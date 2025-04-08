#!/bin/bash
#SBATCH --job-name=embednormeval
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0,3              # job array with 100 jobs (matching num_ranges in script)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jorrel@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2024.6
conda activate plm

python /home/jr8867/embedding-homolog-detection/emb-contr-head/eval-refined-emb.py $SLURM_ARRAY_TASK_ID
