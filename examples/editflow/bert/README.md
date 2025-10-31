
```shell

PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/editflow/bert/pt.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --max_length 128 \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/EditFlow/ModernBERT-large/tiny-shakespeare"

```


```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 8 \
    examples/editflow/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/EditFlow/ModernBERT-large/alpaca"
```