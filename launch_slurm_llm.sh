#!/bin/sh -l
#SBATCH --partition=gpu
#SBATCH --gpus-per-node 4
#SBATCH -c 24
#SBATCH -t 03:00:00 
#SBATCH -N 1
#SBATCH --export=ALL
#SBATCH --job-name=llm-4GPU
#SBATCH --output=%x-%j.out

# get host name
hosts_file="hosts.txt"
# scontrol show hostname $SLURM_JOB_NODELIST > $hosts_file

# Collect public key and accept them
while read -r node; do
    ssh-keyscan "$node" >> ~/.ssh/known_hosts
done < "$hosts_file"

# Create the host file containing node names and the number of GPUs
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=4 if $slots==0;
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile


source /work/projects/ulhpc-tutorials/PS10-Horovod/env_ds.sh


# Launch HuggingFace+DeepSpeed code by varying the number of GPUs
num_gpu=4
# Launch that with a varying batch size

for i in $(seq 1 15); do
    num=$((i * num_gpu))
    output_file="output_LLM_${num_gpu}_${i}.py"
    temp_script="temp_script_LLM_${num_gpu}_${i}.py"

    # Read the content of LLM.py
    cat "LLM.py" > "$output_file"

    # Perform the substitution and write to the new file
    sed "s/^bs = BATCH_SIZE/bs = $num/" "$output_file" > "$temp_script"
    deepspeed --num_gpus $num_gpu --num_nodes 1 --hostfile hostfile $temp_script  # 2 gpu/per node = 2GPUs
done
