# BloomCat: 一套完整的LLM训练以及训练环境
![项目图片](pics/cat.png)

## 项目简介
**BloomCat**是一个基于bloom的开源大语言模型项目，旨在为广大学习
大语言模型的同学提供完整的学习条件，包括数据、模型代码以及运行环境，并发挥广大同学的智慧一起建立完整的大语言模型学习生态，为中文大语言模型的建设添砖加瓦。


##功能模块
- :cat:训练数据
- :cat:基于bloom的预训练
- :cat:基于bloom做lora的finetune
- :cat:基于bloom做qlora的finetune
- :cat:基于bloom的reward model训练
- :cat:基于bloom的ppo
- :cat:数据清洗流程


##训练数据
当前为了验证跑通流程，预训练和finetune的训练数据量还比较少，预训练的数据为悟道开源的200G数据，SFT训练数据为belle的1.5M数据。
- [悟道预训练数据](https://openi.pcl.ac.cn/BAAI/WuDao-Data)
- [belle SFT数据](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)

## 基于bloom的预训练
目前的开源的大多项目，大部分是以加载预训练模型的方式上进行二次的预训练。本项目为了从方便大家学习，完整复现了bloom的
模型结构，并实现了基于deepspeed的流水线并行训练训练方法, 可以直接运行脚本[scripts/pretrain.sh](./scripts/pretrain.sh),
目前是在40G A100上运行, 环境配置见requirements_bloom_pretrain.txt。
```buildoutcfg
deepspeed --num_gpus=8 src/bloom_pretrain.py  --model_config_file bloom-1b/model.config.json --config_pretrain_file run_config/config_pretrain.json --deepspeed run_config/deepspeed_config_stage2.json
```

##基于bloom做lora的finetune
基于lora的训练，实现都是在v100操作。lora的预测版本要求peft的版本不能太高，我配置为0.2.0。
```buildoutcfg
torchrun --nproc_per_node=8 src/finetune.py --model_config_file run_config/Bloom_config.json --lora_hyperparams_file run_config/lora_hyperparams_bloom.json --use_lora True
```

##基于bloom做qlora的finetune
qlora的训练要求peft的版本要最新版，不然会遇到各种错误，我按照的版本为peft==0.4.0.dev0。在v100运行没问题。
```buildoutcfg
python src/finetune_qlora.py --model_config_file run_config/Bloom_config.json --lora_hyperparams_file run_config/lora_hyperparams_bloom.jso
```




