#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=4G
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

export PYTHONPATH=.
python -u train.py        # -u flushes output buffer immediately
