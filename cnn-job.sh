#!/bin/bash
#SBATCH --output=out/ace-job.out #SBATCH --error=out/errors.err
#SBATCH --gres=gpu:2
#SBATCH --mem 64G
#SBATCH -t 1:30:00

module load cudnn
module load cuda90/toolkit
module load cuda90/blas
module load cuda90/profiler
module load cuda90/nsight
module load cuda90/fft
source ~/SGDvm/bin/activate
python DogClassifier.py
