#!/bin/bash

# Prompt the user to enter the batch size (default is 16)
echo "Please enter the batch size (default is 16):"
read batch_size
batch_size=${batch_size:-16}  # If no input, default to batch size 16

# Prompt the user to enter the output directory
echo "Please enter the output directory (default is 'output'):"
read dir
dir=${dir:-output}  # If no input, default to 'output'

timestamp=$(date +"%Y%m%d_%H%M")

output_dir="./nohup/$batch_size/$dir"
output_file="${output_dir}/nohup_${timestamp}.out"
error_log="${output_dir}/error.log"

mkdir -p "$output_dir"

nohup bash fast_run.sh $batch_size $dir > "$output_file" 2> "$error_log" &