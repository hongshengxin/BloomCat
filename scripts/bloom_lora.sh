torchrun --nproc_per_node=8 src/finetune.py --model_config_file run_config/Bloom_config.json --lora_hyperparams_file run_config/lora_hyperparams_bloom.json --use_lora True
