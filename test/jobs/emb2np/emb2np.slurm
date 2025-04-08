#!/bin/bash
#SBATCH --job-name=emb2np
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jorrel@princeton.edu

module purge
module load anaconda3/2024.6
conda activate plm

python /home/jr8867/embedding-homolog-detection/emb-contr-head/emb2np.py
