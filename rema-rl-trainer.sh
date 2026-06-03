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

COMMAND="unset ROCR_VISIBLE_DEVICES;python3 -m pip install --force-reinstall math-verify;python3 -m pip install --force-reinstall --no-deps antlr4-python3-runtime==4.9.3;export PYTHONPATH=/root/ReMA-public/src:/verl:\$PYTHONPATH;python3 -m verl.rema_separated_trainer.main_ppo --config-path=/home/ajanz/projects/ReMA-public/config --config-name=rema-rl.yaml actor_rollout_ref.model.path=${MODEL_PATH}"

srun apptainer exec --nv --writable-tmpfs \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    --mount type=bind,src=$TMPDIR/verl,dst=/verl \
    --mount type=bind,src=$TMPDIR/verl,dst=/root/ReMA-public/src/verl \
    $TMPDIR/verl-rema-v3.sif \
    bash -c "$COMMAND"

if [[ -n $TMPDIR ]]; then
    rm -rf $TMPDIR/*
fi
