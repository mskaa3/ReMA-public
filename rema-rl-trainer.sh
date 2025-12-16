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

rclone copy s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/verl-rema.sif $TMPDIR/
rclone copy s3v2:s3min-tomasznaskret-1712063354/user/dmotyka/sif_images/verl-rema-v2.sif $TMPDIR/

cp ../verl.sif $TMPDIR

source ./env.sh
export HF_HOME=$TMPDIR/hf_home

# COMMAND="unset ROCR_VISIBLE_DEVICES;python3 -m pip install flash-attn==2.7.4.post1 --no-build-isolation;python3 -m pip install -r requirements.txt;python3 -m pip install 'ray==2.10.0';cd src/verl && pip install -e .;python3 -m verl.trainer.main_ppo --config-path=/home/ajanz/projects/ReMA-public/config --config-name=rema-rl.yaml"
COMMAND="unset ROCR_VISIBLE_DEVICES;cd src/verl && pip install -e .;python3 -m verl.trainer.main_ppo --config-path=/home/ajanz/projects/ReMA-public/config --config-name=rema-rl.yaml"

srun apptainer exec --nv \
    --mount type=bind,src=$TMPDIR,dst=$TMPDIR \
    --mount type=bind,src=$TMPDIR,dst=/root/tmpdir \
    $TMPDIR/verl-rema-v2.sif \
    bash -c "$COMMAND"

if [[ -n $TMPDIR ]]; then
    rm -rf $TMPDIR/*
fi