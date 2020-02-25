#!/bin/bash
#SBATCH -n 2                    # two cores
#SBATCH --mem=3G
#SBATCH --time=48:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u train.py        # -u flushes output buffer immediately
