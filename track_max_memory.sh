#!/bin/bash

# Display all processes using GPU
while true; do
    echo "Current GPU processes: (PID, Used Memory, Process name)"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
    echo ""

    read -p "Do you want to check again? (y/n): " answer
    if [[ "$answer" != "n" ]]; then
        echo "Exiting..."
        break
    fi
done

# Prompt the user to enter the PID to monitor
read -p "Please enter the PID to monitor: " PID

MAX_MEMORY=0
MAX_ATTEMPTS=10  # Maximum attempts before exiting
attempts=0
declare -a MEMORY_HISTORY  # Array to store memory usage history
CURRENT_MEMORY_GB=0
counter=0  # Initialize a counter

while true; do
    # Get the current used memory for the specified PID
    CURRENT_MEMORY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -E "^${PID}," | awk -F ', ' '{print $2}' | awk '{print $1}' | sed 's/MiB//')  # Remove unit

    # If the PID is not found
    if [ -z "$CURRENT_MEMORY" ]; then
        attempts=$((attempts + 1))
        if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
            echo "PID $PID not found after $MAX_ATTEMPTS attempts. Exiting."
            exit 1
        fi
        sleep 1
        continue
    fi

    # Convert memory from MiB to GB
    CURRENT_MEMORY_GB=$((CURRENT_MEMORY / 1024))
    CURRENT_MEMORY_REMAINDER=$((CURRENT_MEMORY % 1024))

    # Calculate the fractional part manually
    if [ "$CURRENT_MEMORY_REMAINDER" -ne 0 ]; then
        FRACTION=$((CURRENT_MEMORY_REMAINDER * 100 / 1024))
        CURRENT_MEMORY_GB="${CURRENT_MEMORY_GB}.${FRACTION}"
    else
        CURRENT_MEMORY_GB="${CURRENT_MEMORY_GB}.00"
    fi

    # Store current memory usage in history
    MEMORY_HISTORY+=("$CURRENT_MEMORY_GB")

    # Limit the history to the last 10 seconds (10 entries)
    if [ "${#MEMORY_HISTORY[@]}" -gt 10 ]; then
        MEMORY_HISTORY=("${MEMORY_HISTORY[@]:1}")  # Remove the oldest entry
    fi

    # Increment the counter
    counter=$((counter + 1))

    # Print the current memory usage and history every 10 seconds
    if [ "$counter" -eq 10 ]; then
        echo -e "Memory usage history (last 10 seconds): \n${MEMORY_HISTORY[*]} GB"
        counter=0  # Reset the counter
    fi

    # Update the maximum memory usage without using bc
    MAX_MEMORY_INT=${MAX_MEMORY//.*}  # Get integer part of MAX_MEMORY
    CURRENT_MEMORY_INT=${CURRENT_MEMORY_GB//.*}  # Get integer part of CURRENT_MEMORY_GB

    if [ "$CURRENT_MEMORY_INT" -gt "$MAX_MEMORY_INT" ]; then
        MAX_MEMORY=$CURRENT_MEMORY_GB
        echo "-------------##  New max memory: $MAX_MEMORY GB  ##-------------"
    fi

    sleep 1  # Check every second
done