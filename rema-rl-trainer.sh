#!/bin/bash
#SBATCH --job-name=verl-trainer     # nazwa
#SBATCH --nodes=1                   # ilość węzłów
#SBATCH --cpus-per-gpu=4            # ilość cpu na zadanie
#SBATCH --time=48:00:00             # maksymalny czas wykonania zadania
#SBATCH --mem=200gb                 # ilość pamięci RAM
#SBATCH -p lem-gpu-short            # partycja
#SBATCH --gpus-per-node=hopper:4    # (ilość kart graficznych na węźle)
#SBATCH --verbose                   # wyświetlanie informacji o zadaniu

cp -r ./data/overall_math $TMPDIR/overall_math
cp -r ./data/MATH $TMPDIR/MATH
cp -r ./src/verl $TMPDIR/verl

rclone copy s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/verl-rema-v3.sif $TMPDIR/

source ./env.sh
export HF_HOME=$TMPDIR/hf_home
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
PRD_EXPORT_ENABLE=${PRD_EXPORT_ENABLE:-0}
PRD_EXPORT_MAX_RECORDS=${PRD_EXPORT_MAX_RECORDS:-50000}
PRD_EXPORT_LOCAL=${PRD_EXPORT_LOCAL:-$TMPDIR/prd_reward_composer/prd_records_${SLURM_JOB_ID}.jsonl}
PRD_EXPORT_REMOTE=${PRD_EXPORT_REMOTE:-}
PRD_ONLINE_ENABLE=${PRD_ONLINE_ENABLE:-1}
PRD_ONLINE_WARMUP_STEPS=${PRD_ONLINE_WARMUP_STEPS:-200}
PRD_ONLINE_BLEND_ALPHA=${PRD_ONLINE_BLEND_ALPHA:-0.05}
PRD_ONLINE_BLEND_ALPHA_MAX=${PRD_ONLINE_BLEND_ALPHA_MAX:-0.05}
PRD_ONLINE_BLEND_RAMP_STEPS=${PRD_ONLINE_BLEND_RAMP_STEPS:-200}
PRD_ONLINE_SYNC_INTERVAL=${PRD_ONLINE_SYNC_INTERVAL:-50}
PRD_ONLINE_EMA_BETA=${PRD_ONLINE_EMA_BETA:-0.9}
ROLLOUT_N=${ROLLOUT_N:-32}

PRD_EXPORT_OVERRIDES=""
if [[ "${PRD_EXPORT_ENABLE}" == "1" || "${PRD_EXPORT_ENABLE}" == "true" ]]; then
    PRD_EXPORT_OVERRIDES="algorithm.hierarchy.reward_composer.export_jsonl_path=${PRD_EXPORT_LOCAL} algorithm.hierarchy.reward_composer.export_jsonl_max_records=${PRD_EXPORT_MAX_RECORDS}"
    echo "PRD reward-composer export enabled: ${PRD_EXPORT_LOCAL}"
fi

PRD_ONLINE_OVERRIDES="algorithm.hierarchy.reward_composer.enable=False actor_rollout_ref.rollout.n=${ROLLOUT_N}"
if [[ "${PRD_ONLINE_ENABLE}" == "1" || "${PRD_ONLINE_ENABLE}" == "true" ]]; then
    PRD_ONLINE_OVERRIDES="${PRD_ONLINE_OVERRIDES} algorithm.hierarchy.reward_composer.online.enable=True algorithm.hierarchy.reward_composer.online.warmup_steps=${PRD_ONLINE_WARMUP_STEPS} algorithm.hierarchy.reward_composer.online.blend_alpha=${PRD_ONLINE_BLEND_ALPHA} algorithm.hierarchy.reward_composer.online.blend_alpha_max=${PRD_ONLINE_BLEND_ALPHA_MAX} algorithm.hierarchy.reward_composer.online.blend_ramp_steps=${PRD_ONLINE_BLEND_RAMP_STEPS} algorithm.hierarchy.reward_composer.online.mixed_groups_only=True algorithm.hierarchy.reward_composer.online.sync_interval=${PRD_ONLINE_SYNC_INTERVAL} algorithm.hierarchy.reward_composer.online.ema_beta=${PRD_ONLINE_EMA_BETA}"
    echo "Online PRD reward composer enabled with rollout.n=${ROLLOUT_N}, alpha=${PRD_ONLINE_BLEND_ALPHA}, warmup=${PRD_ONLINE_WARMUP_STEPS}"
else
    PRD_ONLINE_OVERRIDES="${PRD_ONLINE_OVERRIDES} algorithm.hierarchy.reward_composer.online.enable=False"
    echo "Online PRD reward composer disabled; rollout.n=${ROLLOUT_N}"
fi

COMMAND="unset ROCR_VISIBLE_DEVICES;python3 -m pip install --force-reinstall math-verify;python3 -m pip install --force-reinstall --no-deps antlr4-python3-runtime==4.9.3;export PYTHONPATH=/root/ReMA-public/src:/verl:\$PYTHONPATH;python3 -m verl.rema_separated_trainer.main_ppo --config-path=/home/ajanz/projects/ReMA-public/config --config-name=rema-rl.yaml actor_rollout_ref.model.path=${MODEL_PATH} trainer.n_gpus_per_node=2 trainer.val_before_train=False trainer.test_freq=50 actor_rollout_ref.rollout.max_num_batched_tokens=16384 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 algorithm.hierarchy.num_worker_stages=3 ${PRD_ONLINE_OVERRIDES} ${PRD_EXPORT_OVERRIDES}"

srun apptainer exec --nv --writable-tmpfs \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    --mount type=bind,src=$TMPDIR/verl,dst=/verl \
    --mount type=bind,src=$TMPDIR/verl,dst=/root/ReMA-public/src/verl \
    $TMPDIR/verl-rema-v3.sif \
    bash -c "$COMMAND"

if [[ ("${PRD_EXPORT_ENABLE}" == "1" || "${PRD_EXPORT_ENABLE}" == "true") && -n "${PRD_EXPORT_REMOTE}" ]]; then
    if [[ -s "${PRD_EXPORT_LOCAL}" ]]; then
        gzip -c "${PRD_EXPORT_LOCAL}" > "${PRD_EXPORT_LOCAL}.gz"
        rclone copyto "${PRD_EXPORT_LOCAL}.gz" "${PRD_EXPORT_REMOTE}/prd_records_${SLURM_JOB_ID}.jsonl.gz"
        echo "Uploaded PRD export to ${PRD_EXPORT_REMOTE}/prd_records_${SLURM_JOB_ID}.jsonl.gz"
    else
        echo "No PRD export found at ${PRD_EXPORT_LOCAL}; skipping upload"
    fi
fi

if [[ -n $TMPDIR ]]; then
    rm -rf $TMPDIR/*
fi
