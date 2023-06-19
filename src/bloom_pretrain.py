import os
os.environ["WANDB_DISABLED"] = "true"
import sys
from typing import List
import argparse, logging

import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers
from models.modelling_bloom import BloomForCausalLM
from bloom_dataset_len import DataList
from accelerate import init_empty_weights, infer_auto_device_map
import json
import deepspeed


#assert (
#    "LlamaTokenizer" in transformers._import_structure["models.llama"]
#), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import PretrainedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_logger(logger_name,output_dir):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) 
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir,'log.txt'),mode='w')
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(console_handler)
    return logger
 


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train(
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = True,  # faster, but produces an odd training loss curve,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):

    reptrain_config = json.load(open(args.config_pretrain_file))
    model_type = reptrain_config['model_type']
    model_name_or_path = reptrain_config['model_name_or_path']
    data_path = reptrain_config['data_path']
    output_dir = reptrain_config['output_dir']
    cutoff_len = reptrain_config['cutoff_len']

    print(reptrain_config)

    deepspeed.init_distributed()

    logger = get_logger("train", reptrain_config['output_dir'])
    logger.info("args.__dict__ : {}".format(args.__dict__))

    for key, value in reptrain_config.items():
        logger.info("{} : {}".format(key, value))
    assert (
        model_name_or_path
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    gradient_accumulation_steps = reptrain_config['batch_size'] // reptrain_config['per_device_train_batch_size'] if "gradient_accumulation_steps" not in reptrain_config else reptrain_config['gradient_accumulation_steps']
    print(gradient_accumulation_steps)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    load_in_8bit = True if args.use_lora else False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = "left" ## 待实验

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= cutoff_len:
            result["input_ids"][cutoff_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][cutoff_len - 1] = 1

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        
        try:
            input_text = data_point["content"]
            input_text = tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text
            input_text = input_text + tokenizer.eos_token
            full_prompt = input_text
            tokenized_full_prompt = tokenize(full_prompt)

            if not train_on_inputs:
                user_prompt = input_text
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ] 
            
            return tokenized_full_prompt
        except:
            pass

    
    data_files = {'train': data_path}

    data = load_dataset("json", data_files=data_path, split='train', streaming=True)
    # tmp = next(iter(data))
    # print(tmp)
    # print(generate_and_tokenize_prompt(tmp))
    # exit(0)
    val_set_size = reptrain_config['val_set_size']
    if val_set_size > 0:
        val_set_size = min(val_set_size, int(len(data['train'])*reptrain_config['val_set_rate']))
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    val_dataset = None

    if args.use_lora:
        model = prepare_model_for_int8_training(model)
        lora_hyperparams = json.load(open(args.lora_hyperparams_file))
        for key, value in lora_hyperparams.items():
            logger.info("{} : {}".format(key, value))
        config = LoraConfig(
            r=lora_hyperparams['lora_r'],
            lora_alpha=lora_hyperparams['lora_alpha'],
            target_modules=lora_hyperparams['lora_target_modules'] if model_config['model_type']=="Llama" else ["query_key_value"],
            lora_dropout=lora_hyperparams['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(config)
        model = get_peft_model(model, config)

    model_config = PretrainedConfig.from_pretrained(args.model_config_file)
    if model_type.lower() == "bloom":
        load_in_8bit = True if args.use_lora else False
        model = BloomForCausalLM(model_config)
        # device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16")

        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name_or_path,
        #     load_in_8bit = load_in_8bit,
        #     device_map=device_map,
        # )
    model.half()
    print(model)
    print(get_parameter_number(model))
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    val_set_size = reptrain_config['val_set_size']
    print("start train...")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=reptrain_config['per_device_train_batch_size'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=reptrain_config['warmup_steps'],
            num_train_epochs=reptrain_config['num_epochs'],
            learning_rate=reptrain_config['learning_rate'],
            fp16=True,
            logging_steps=reptrain_config['logging_steps'],
            logging_dir="tensorboard",
            logging_strategy="steps",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=reptrain_config["eval_steps"] if val_set_size > 0 else None,
            save_steps=reptrain_config["save_steps"],
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=args.deepspeed if not args.use_lora else None,
            # group_by_length=group_by_length,
            max_steps = 10000000,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    print("trainer.train")
    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)
    logger.info("Save checkpointing...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above when using lora to train, please disregard :)")
    logger.info("Training succeeded")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--config_pretrain_file", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, help="deepspeed config")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--lora_hyperparams_file", default="", type=str, help="Provide it when use_lora=True")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use lora")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    fire.Fire(train)
