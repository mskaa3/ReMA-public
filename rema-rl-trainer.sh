#!/bin/bash
#SBATCH --job-name=verl-trainer     # nazwa
#SBATCH --nodes=1                   # ilość węzłów
#SBATCH --cpus-per-gpu=4            # ilość cpu na zadanie
#SBATCH --time=72:00:00             # maksymalny czas wykonania zadania
#SBATCH --mem=200gb                 # ilość pamięci RAM
#SBATCH -p lem-gpu-short            # partycja
#SBATCH --gpus-per-node=hopper:2    # (ilość kart graficznych na węźle)
#SBATCH --verbose                   # wyświetlanie informacji o zadaniu

cp -r ./data/overall_math $TMPDIR/overall_math
cp -r ./data/MATH $TMPDIR/MATH

rclone copy s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/verl-rema-v3.sif $TMPDIR/

source ./env.sh
export HF_HOME=$TMPDIR/hf_home

COMMAND="unset ROCR_VISIBLE_DEVICES;python3 -m verl.rema_trainer.main_ppo --config-path=/home/ajanz/projects/ReMA-public/config --config-name=rema-rl.yaml"

srun apptainer exec --nv \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    $TMPDIR/verl-rema-v3.sif \
    bash -c "$COMMAND"

if [[ -n $TMPDIR ]]; then
    rm -rf $TMPDIR/*
fi