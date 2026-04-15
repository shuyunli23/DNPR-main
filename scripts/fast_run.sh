#!/bin/bash
#===============================================================================
# DNPR Fast Evaluation Script
# 
# Runs evaluation on multiple datasets with configurable parameters.
#
# Usage:
#   bash scripts/fast_run.sh [OPTIONS]
#
# Examples:
#   bash scripts/fast_run.sh                           # Use all defaults
#   bash scripts/fast_run.sh -b 32 -g 0                # Batch size 32, GPU 0
#   bash scripts/fast_run.sh -d "mvtec visa"           # Only MVTec and VisA
#   bash scripts/fast_run.sh -n 10 -k 1                # 10 runs, 1-shot
#   bash scripts/fast_run.sh --datasets mvtec --gpu 2  # Long form options
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# Default Configuration
#-------------------------------------------------------------------------------
BATCH_SIZE=16
OUTPUT_DIR="output"
NUM_RUNS=5
GPU=1
K_SHOT=0
NBR=9
GLO_MEMORY=12
LOC_MEMORY=3
K_MIN=0.05
BACKBONE="wideresnet50"
DATASETS=""  # Empty means use default dataset list
CONFIG_DIR="./configs"
DRY_RUN=false

#-------------------------------------------------------------------------------
# Color Output
#-------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#-------------------------------------------------------------------------------
# Help Message
#-------------------------------------------------------------------------------
show_help() {
    cat << EOF
DNPR Fast Evaluation Script

Usage: bash scripts/fast_run.sh [OPTIONS]

Options:
  -b, --batch-size SIZE     Batch size (default: ${BATCH_SIZE})
  -o, --output DIR          Output directory name (default: ${OUTPUT_DIR})
  -n, --num-runs N          Number of runs per dataset (default: ${NUM_RUNS})
  -g, --gpu ID              GPU ID to use (default: ${GPU})
  -k, --k-shot K            K-shot value, 0 for zero-shot (default: ${K_SHOT})
  -d, --datasets "D1 D2"    Space-separated dataset list (default: mvtec visa bt ci)
      --nbr SIZE            Neighborhood size (default: ${NBR})
      --gm SIZE             Global memory bank size (default: ${GLO_MEMORY})
      --lm SIZE             Local memory bank size (default: ${LOC_MEMORY})
      --k-min VALUE         Minimum k value (default: ${K_MIN})
      --backbone NAME       Backbone network (default: ${BACKBONE})
      --config-dir DIR      Config directory (default: ${CONFIG_DIR})
      --dry-run             Print commands without executing
  -h, --help                Show this help message

Available Datasets:
  mvtec   - MVTec AD Dataset
  visa    - VisA Dataset
  bt      - BTAD Dataset
  ci      - CableInspect-AD Dataset
  dtd     - DTD-Synthetic Dataset
  rad     - RAD Dataset

Available Backbones:
  wideresnet50, wideresnet101, resnet50, resnet101,
  resnet18, resnext101

Examples:
  # Run with default settings
  bash scripts/fast_run.sh

  # Custom batch size and GPU
  bash scripts/fast_run.sh -b 32 -g 0

  # Specific datasets only
  bash scripts/fast_run.sh -d "mvtec visa"

  # Few-shot evaluation
  bash scripts/fast_run.sh -k 4 -n 3

  # Full customization
  bash scripts/fast_run.sh -b 8 -g 2 -n 10 -d "mvtec" --backbone resnet50

EOF
    exit 0
}

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -k|--k-shot)
            K_SHOT="$2"
            shift 2
            ;;
        -d|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        --nbr)
            NBR="$2"
            shift 2
            ;;
        --gm)
            GLO_MEMORY="$2"
            shift 2
            ;;
        --lm)
            LOC_MEMORY="$2"
            shift 2
            ;;
        --k-min)
            K_MIN="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Validate Inputs
#-------------------------------------------------------------------------------
validate_positive_int() {
    local name=$1
    local value=$2
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
        log_error "$name must be a positive integer (got: $value)"
        exit 1
    fi
}

validate_non_negative_int() {
    local name=$1
    local value=$2
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        log_error "$name must be a non-negative integer (got: $value)"
        exit 1
    fi
}

validate_float() {
    local name=$1
    local value=$2
    if ! [[ "$value" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        log_error "$name must be a number (got: $value)"
        exit 1
    fi
}

validate_positive_int "Batch size" "$BATCH_SIZE"
validate_positive_int "Number of runs" "$NUM_RUNS"
validate_non_negative_int "GPU ID" "$GPU"
validate_non_negative_int "K-shot" "$K_SHOT"
validate_positive_int "Neighborhood size" "$NBR"
validate_positive_int "Global memory size" "$GLO_MEMORY"
validate_positive_int "Local memory size" "$LOC_MEMORY"
validate_float "K-min" "$K_MIN"

#-------------------------------------------------------------------------------
# Set Default Datasets if Not Specified
#-------------------------------------------------------------------------------
if [[ -z "$DATASETS" ]]; then
    DATASETS="mvtec visa bt ci"
fi

# Convert string to array
read -ra DATASET_ARRAY <<< "$DATASETS"

#-------------------------------------------------------------------------------
# Get Config File Path
#-------------------------------------------------------------------------------
get_config_path() {
    local dataset=$1
    case "$dataset" in
        mvtec)
            echo "${CONFIG_DIR}/mvtec.yaml"
            ;;
        visa)
            echo "${CONFIG_DIR}/visa.yaml"
            ;;
        bt|btad)
            echo "${CONFIG_DIR}/btad.yaml"
            ;;
        ci|ciad)
            echo "${CONFIG_DIR}/ciad.yaml"
            ;;
        dtd|dtd_synthetic)
            echo "${CONFIG_DIR}/dtd_synthetic.yaml"
            ;;
        rad)
            echo "${CONFIG_DIR}/rad.yaml"
            ;;
        *)
            log_error "Unknown dataset: $dataset"
            exit 1
            ;;
    esac
}

#-------------------------------------------------------------------------------
# Print Configuration
#-------------------------------------------------------------------------------
print_config() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    DNPR Fast Evaluation                        ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}Datasets:${NC}        ${DATASET_ARRAY[*]}"
    echo -e "  ${GREEN}Batch size:${NC}      $BATCH_SIZE"
    echo -e "  ${GREEN}GPU:${NC}             $GPU"
    echo -e "  ${GREEN}K-shot:${NC}          $K_SHOT"
    echo -e "  ${GREEN}Num runs:${NC}        $NUM_RUNS"
    echo -e "  ${GREEN}Output dir:${NC}      $OUTPUT_DIR"
    echo ""
    echo -e "  ${GREEN}Backbone:${NC}        $BACKBONE"
    echo -e "  ${GREEN}Neighborhood:${NC}    $NBR"
    echo -e "  ${GREEN}Global memory:${NC}   $GLO_MEMORY"
    echo -e "  ${GREEN}Local memory:${NC}    $LOC_MEMORY"
    echo -e "  ${GREEN}K-min:${NC}           $K_MIN"
    echo ""
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "  ${YELLOW}Mode:${NC}            DRY RUN (commands will not be executed)"
        echo ""
    fi
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

#-------------------------------------------------------------------------------
# Run Command
#-------------------------------------------------------------------------------
run_command() {
    local cmd=$1
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $cmd"
    else
        echo -e "${GREEN}[EXEC]${NC} $cmd"
        eval "$cmd"
    fi
}

#-------------------------------------------------------------------------------
# Main Execution
#-------------------------------------------------------------------------------
print_config

total_datasets=${#DATASET_ARRAY[@]}
current_dataset=0

for dataset in "${DATASET_ARRAY[@]}"; do
    ((++current_dataset))
    
    # Get config path
    cfg=$(get_config_path "$dataset")
    
    # Check if config exists
    if [[ ! -f "$cfg" ]] && [[ "$DRY_RUN" == false ]]; then
        log_warn "Config file not found: $cfg, skipping $dataset"
        continue
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Dataset [$current_dataset/$total_datasets]: ${dataset^^}${NC}"
    echo -e "${BLUE}  Config: $cfg${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    for ((seed=0; seed<NUM_RUNS; seed++)); do
        run_num=$((seed + 1))
        log_info "Run $run_num/$NUM_RUNS (seed=$seed)"
        
        # Build command
        cmd="python -m dnpr.main"
        cmd+=" --seed $seed"
        cmd+=" --gpu $GPU"
        cmd+=" -k $K_SHOT"
        cmd+=" --cfg $cfg"
        cmd+=" --batch_size $BATCH_SIZE"
        cmd+=" -km $K_MIN"
        cmd+=" --nbr $NBR"
        cmd+=" -gm $GLO_MEMORY"
        cmd+=" -lm $LOC_MEMORY"
        cmd+=" --backbone $BACKBONE"
        cmd+=" --resume $OUTPUT_DIR"
        
        # Add aggregate metrics flag on last run
        if (( seed == NUM_RUNS - 1 )); then
            cmd+=" -am $NUM_RUNS"
        fi
        
        run_command "$cmd"
        echo ""
    done
done

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Evaluation Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Results saved to: ${GREEN}results/$OUTPUT_DIR${NC}"
echo ""