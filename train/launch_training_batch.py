import argparse
import concurrent.futures
import os
import random
import subprocess
import time


def create_train_script_deepspeed(
    dset_name,
    num_gpu,
    num_node,
    gradient_accum_step,
    per_dev_batch,
    seed,
    lr,
    train_size,
    epoch,
    folder,
    ds_config,
    mem,
    gpu,
    partition,
    include,
    base_model_path,
    template,
    warm_up,
):
    batch_size = num_gpu * per_dev_batch * gradient_accum_step
    k = int(train_size) // 1000
    rand_int = random.randint(0, 9)
    base_model_path_lines = base_model_path.split("/")
    for i in range(len(base_model_path_lines)):
        if (
            i < len(base_model_path_lines) - 1
            and base_model_path_lines[i] == "ckpts"
        ):
            model_name = base_model_path_lines[i + 1]
            break
    train_string = f"""#!/bin/bash

#SBATCH --job-name=cl_{dset_name}_{lr}_b{batch_size}s{seed}
#SBATCH --output=cl_{dset_name}_{lr}_b{batch_size}s{seed}.out
#SBATCH --error=cl_{dset_name}_{lr}_b{batch_size}s{seed}.err

#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:{gpu}:{num_gpu}
#SBATCH --mem={mem}
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL

source ~/.bashrc
conda activate synatra

WANDB__SERVICE_WAIT=500 WANDB_PROJECT=WANDB_PROJECT WANDB_ENTITY=WANDB_ENTITY WANDB_NAME={dset_name}_b{batch_size}s{seed} deepspeed --hostfile hostfile --include {include} --master_port=998{rand_int} ./LLaMA-Factory-0.8.3/src/train.py \
    --deepspeed {ds_config} \
    --stage sft \
    --model_name_or_path {base_model_path} \
    --do_train \
    --val_size 2 \
    --dataset {dset_name} \
    --dataset_dir DATASET_DIR/{folder}/ \
    --tokenized_path CACHE_PATH/{dset_name}_{model_name}/ \
    --template {template} \
    --finetuning_type full \
    --output_dir OUTPUT_DIR/output2_{k}k_{dset_name}_{model_name}_{lr}_b{batch_size}s{seed}/ \
    --overwrite_output_dir True \
    --save_only_model \
    --per_device_train_batch_size {per_dev_batch} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accum_step} \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 500 \
    --logging_steps "30" \
    --save_total_limit 8 \
    --preprocessing_num_workers 16 \
    --learning_rate {lr} \
    --weight_decay 0. \
    {warm_up} \
    --num_train_epochs {epoch} \
    --plot_loss \
    --bf16 True \
    --cutoff_len 4096 \
    --report_to 'wandb' \
    --flash_attn fa2 \
    --seed {seed}

echo "exit code: $?"

    """
    with open(
        f"./train_script/tmp_{dset_name}_{model_name}_{seed}.sh", "w"
    ) as file:
        file.write(train_string)
    return


dset = [
    ["YOUR_TRAINING_SET", 99920],
]

for seed in [87]:
    for dset_name, train_size in dset:
        dset_name = dset_name
        num_gpu = 4
        num_node = 1
        partition = "general"
        include = "YOUR_GPU_SLOTS"
        gpu = "A100"
        per_dev_batch = 6
        gradient_accum_step = 2
        lr = "4e-5"
        warm_up = "--warmup_ratio 0.03"
        seed = seed
        train_size = train_size
        folder = "code"
        epoch = 5
        mem = "1000G"
        ds_config = "YOUR_DEEPSPEED_CONFIG"
        base_model_path = "YOUR_BASE_MODEL_PATH"
        template = "llama2"
        # template = 'llama3'
        accelerate_config = "accelerate_single_config.yaml"
        create_train_script_deepspeed(
            dset_name,
            num_gpu,
            num_node,
            gradient_accum_step,
            per_dev_batch,
            seed,
            lr,
            train_size,
            epoch,
            folder,
            ds_config,
            mem,
            gpu,
            partition,
            include,
            base_model_path,
            template,
            warm_up,
        )
        subprocess.run(["sbatch", f"./train_script/tmp_{dset_name}_{seed}.sh"])
