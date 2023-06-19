deepspeed --num_gpus=8 src/bloom_pretrain.py  --model_config_file bloom-1b/model.config.json --config_pretrain_file run_config/config_pretrain.json --deepspeed run_config/deepspeed_config_stage2.json

