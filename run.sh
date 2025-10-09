#!/bin/bash

# Prompt the user to enter the number of executions
echo "Please enter the number of executions: (default is 10)"
read num_times
num_times=${num_times:-10}  # If no input, default to 10

# Validate if the input is a positive integer
if ! [[ "$num_times" =~ ^[0-9]+$ ]] ; then
   echo "Error: The input must be a positive integer."
   exit 1
fi

# Prompt the user to enter the GPU number
echo "Please enter the GPU number (default is 2):"
read gpu
gpu=${gpu:-2}  # If no input, default to GPU 2

# Prompt the user to enter the k-shot value
echo "Please enter the k-shot value (default is 0):"
read k_shot
k_shot=${k_shot:-0}  # If no input, default to k-shot 0

# Prompt the user to choose the config file ('visa', 'bt', default is 'mvtec')
echo "Please enter the config file choice ('visa', 'bt', 'dtd', 'ci', 'rad', default is 'mvtec'):"
read cfg_choice

# Select the config file based on user input
case "$cfg_choice" in
    visa)
        cfg="./datasets/config_visa_dataset.yaml"  # If input is 'visa', use visa config
        ;;
    bt)
        cfg="./datasets/config_bt_dataset.yaml"  # If input is 'bt', use bt config
        ;;
    dtd)
        cfg="./datasets/config_dtd_dataset.yaml"  # If input is 'dtd', use dtd config
        ;;
    rad)
        cfg="./datasets/config_rad_dataset.yaml"  # If input is 'rad', use rad config
        ;;
    ci)
        cfg="./datasets/config_ci_dataset.yaml"  # If input is 'ci', use ci config
        ;;
    *)
        cfg="./datasets/config_dataset.yaml"  # Default is './datasets/config_dataset.yaml'
        ;;
esac

# Prompt the user to enter the size (default is empty)
echo "Please enter the resize dimensions and crop size (resize_width resize_height crop_width crop_height), separated by spaces:"
read -a size_input  # Read input into an array

# Check if the user provided exactly 4 inputs; if not, set default (empty array)
if [ ${#size_input[@]} -ne 4 ]; then
    echo "Invalid input. Defaulting to empty array."
    size_input=()  # Default to empty if the input is not exactly 4 values
else
    # Input is valid, no need to modify size_input
    echo "Input size: ${size_input[@]}"
fi

# Prompt the user to enter the batch size (default is 16)
echo "Please enter the batch size (default is 16):"
read batch_size
batch_size=${batch_size:-16}  # If no input, default to batch size 16

echo "Run $num_times times"
# Loop through the specified number of executions
for ((seed=0; seed<num_times; seed++))
do
    # Construct the command
    command="python main.py --seed $seed --gpu $gpu -k $k_shot --cfg $cfg --batch_size $batch_size"

    # Check if size_input is not empty and append it to the command
    if [ ${#size_input[@]} -ne 0 ]; then
        command+=" -s ${size_input[@]}"
    fi

    # For the last execution, add the -am parameter
    if (( seed == num_times - 1 )); then
        command+=" -am $num_times"
    fi
    echo "Executing run $((seed+1)) with seed=$seed, GPU=$gpu, k-shot=$k_shot, cfg=$cfg, batch_size=$batch_size:
    $command"

    # Execute the command
    eval $command
done
