#!/bin/bash

# Check for batch size argument
if [ -z "$1" ]; then
    batch_size=16  # Default to 16 if not provided
else
    batch_size=$1  # Get from command line
fi

# Check for output directory argument
if [ -z "$2" ]; then
    dir="output"  # Default to 'output' if not provided
else
    dir=$2  # Get from command line
fi

if [ "$dir" == "output" ]; then
    datasets=("mvtec" "visa" "bt" "ci")
else
#    datasets=("mvtec")
    datasets=("mvtec" "visa")
fi

num_times=5
gpu=0; k_shot=0; input_nbr=9; input_gm=12; model="wideresnet50"; input_lm=3

for (( i = 0; i < ${#datasets[@]}; i++ )); do
    echo "Test ${datasets[i]}"

    # Select the config file based on dataset name
    case "${datasets[i]}" in
        visa)
            cfg="./datasets/config_visa_dataset.yaml"  # If input is 'visa', use visa config
            ;;
        bt)
            cfg="./datasets/config_bt_dataset.yaml"  # If input is 'bt', use bt config
            ;;
        ci)
            cfg="./datasets/config_ci_dataset.yaml"  # If input is 'ci', use ci config
            ;;
        *)
            cfg="./datasets/config_dataset.yaml"  # Default config
            ;;
    esac

    echo "Run $num_times times"

    # Loop through the specified number of executions
    for ((seed=0; seed<num_times; seed++))
    do
        # Construct the command
        command="python main.py --seed $seed --gpu $gpu -k $k_shot --cfg $cfg --batch_size $batch_size"

        # For the last execution, add the -am parameter
        if (( seed == num_times - 1 )); then
            command+=" -am $num_times"
        fi
        command+=" --nbr $input_nbr -gm $input_gm --backbone $model --resume $dir -lm $input_lm"

        echo -e "Executing run $((seed+1)) with seed=$seed, k-shot=$k_shot, cfg=$cfg: \n$command"

        # Execute the command
        $command
    done
done