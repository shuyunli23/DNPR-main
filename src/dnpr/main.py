#!/usr/bin/env python3
"""
DNPR: Zero-Shot Industrial Anomaly Detection via Dynamic Normal Prototype Evolution.

Main entry point for training and evaluation.

Usage:
    python -m dnpr.main --config configs/mvtec.yaml --gpu 0 --k_shot 0
"""

import argparse
import copy
import logging
import os
import warnings
from datetime import datetime

import torch
import yaml
from easydict import EasyDict

from dnpr.datasets import (
    CLASS_NAMES,
    TEXTURE_CLASS_NAMES,
    VISA_CLASS_NAMES,
    BT_CLASS_NAMES,
    DTD_CLASS_NAMES,
    CI_CLASS_NAMES,
    RAD_CLASS_NAMES,
    build_custom_dataloader,
    build_generic_dataloader,
    select_training_data,
)
from dnpr.models import DNPR, load
from dnpr.utils import (
    fix_seeds, 
    set_torch_device,
    save_metrics_to_csv, 
    aggregate_metrics
    )

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DNPR: Zero-Shot Industrial Anomaly Detection"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_path", type=str, default="./results", help="Path to save results"
    )
    parser.add_argument(
        "--backbone", type=str, default="wideresnet50", help="Backbone name"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        nargs=4,
        metavar=("resize_w", "resize_h", "crop_w", "crop_h"),
        default=[],
        help="Resize and crop dimensions",
    )
    parser.add_argument(
        "--layers_to_extract_from",
        "-le",
        type=str,
        nargs="+",
        default=["layer2", "layer3"],
        help="Feature extraction layers",
    )
    parser.add_argument("--nbr", type=int, default=3, help="Neighborhood size")
    parser.add_argument(
        "--feat_dim", "-fd", type=int, default=1024, help="Feature dimensions"
    )
    parser.add_argument(
        "--proj_dim", "-pd", type=int, default=256, help="Projection dimensions"
    )
    parser.add_argument(
        "--glo_memory_num", "-gm", type=int, default=12, help="Global memory bank size"
    )
    parser.add_argument(
        "--loc_memory_num", "-lm", type=int, default=3, help="Local memory bank size"
    )
    parser.add_argument(
        "--feat_crop_ratio", "-fcr", type=float, default=0.9, help="Feature crop ratio"
    )
    parser.add_argument("--k_min", "-km", type=float, default=0.05, help="Minimum k")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID")
    parser.add_argument("--k_shot", "-k", type=int, default=0, help="K-shot")
    parser.add_argument("--is_plot", "-plt", action="store_true", help="Enable plotting")
    parser.add_argument(
        "--aggregate_metrics",
        "-am",
        type=int,
        default=-1,
        help="Aggregate metrics from multiple runs",
    )
    parser.add_argument(
        "--resume", type=str, default="exp1st", help="Output directory name"
    )
    parser.add_argument(
        "--padding_mode", "-pm", type=str, default="border", help="Registration padding mode"
    )
    parser.add_argument(
        "--cfg", default="./configs/mvtec.yaml", help="Config file path"
    )
    return parser.parse_args()


def get_class_names_and_strategies(dataset_type: str):
    """Get class names and strategies based on dataset type."""
    if dataset_type == "custom":
        class_names = CLASS_NAMES
        strategies = {
            "screw": "transform",
            "metal_nut": "transform",
            **{name: "remain" for name in CLASS_NAMES if name not in ["screw", "metal_nut"]},
        }
    elif dataset_type == "visa":
        class_names = VISA_CLASS_NAMES
        strategies = {name: "remain" for name in VISA_CLASS_NAMES}
    elif dataset_type == "dtd":
        class_names = DTD_CLASS_NAMES
        strategies = {name: "remain" for name in DTD_CLASS_NAMES}
    elif dataset_type == "rad":
        class_names = RAD_CLASS_NAMES
        strategies = {name: "remain" for name in RAD_CLASS_NAMES}
    elif dataset_type == "ci":
        class_names = CI_CLASS_NAMES
        strategies = {name: "remain" for name in CI_CLASS_NAMES}
    else:  # btad
        class_names = BT_CLASS_NAMES
        strategies = {name: "remain" for name in BT_CLASS_NAMES}

    return class_names, strategies


def run(config: argparse.Namespace) -> None:
    """Main evaluation loop."""
    device = set_torch_device([config.gpu])

    metrics = {}

    print(f"{config.k_shot}-shot anomaly detection using device {config.gpu}.")
    print_message = "All classes: "

    # Load dataset config
    with open(config.cfg) as f:
        config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    dataset_type = config_dataset.dataset["type"]
    class_names, strategies = get_class_names_and_strategies(dataset_type)

    all_class = " ".join(class_names)
    print_message += f"{all_class}\n"

    # Update config
    config_dataset.dataset["batch_size"] = config.batch_size
    config.resume = (
        f"{config.resume}/batch_{config.batch_size}/seed_{config.seed}/"
        f"{config.k_shot}_shot_{dataset_type}[{config.backbone}]"
    )
    save_dir = os.path.join(config.save_path, config.resume)
    os.makedirs(save_dir, exist_ok=True)

    # Print configuration
    print_message += f"Info: | Backbone: {config.backbone} | "
    print_message += f"Layers: {config.layers_to_extract_from} | "
    print_message += f"Batch size: {config.batch_size} | "

    if config.size:
        config_dataset.dataset["input_size"] = config.size[:2]
        config_dataset.dataset["crop_size"] = config.size[2:4]

    config.resize = config_dataset.dataset["input_size"]
    config.crop_size = config_dataset.dataset["crop_size"]
    print_message += f"Resize: {config.resize} | "
    print_message += f"Crop: {config.crop_size} |"
    print(print_message)

    # Prepare dataset configs
    cfg_dataset = config_dataset.dataset
    train_dataset = copy.deepcopy(cfg_dataset)
    cfg_dataset.update(cfg_dataset.get("test", None))
    train_dataset.update(train_dataset.get("train", None))

    # Evaluate each class
    for class_name in class_names:
        if dataset_type == "custom":
            cls_dataloader = build_custom_dataloader(
                cfg_dataset, False, False, class_name=class_name
            )
        else:
            cls_dataloader = build_generic_dataloader(
                cfg_dataset, False, False, class_name=class_name
            )

        info_set = {"dataset_set": cls_dataloader, "cls_name": class_name}

        model = DNPR(
            config,
            load(config.backbone),
            config.layers_to_extract_from,
            info_set,
            device,
            (3, *config.crop_size),
            config.feat_dim,
            config.feat_dim,
            3,
            1,
            proj_dim=config.proj_dim,
            strategy=strategies[class_name],
        )

        if config.k_shot > 0:
            train_data = select_training_data(
                train_dataset, k_shot=config.k_shot, class_name=class_name
            )
            model.few_shot_memory(train_data)

        metrics[class_name] = model.test()
        torch.cuda.empty_cache()

    save_metrics_to_csv(save_dir, metrics)

    # Aggregate metrics if requested
    if config.aggregate_metrics > 0:
        _setup_logging_and_aggregate(config, config_dataset, strategies, save_dir)


def _setup_logging_and_aggregate(
    config: argparse.Namespace,
    config_dataset: EasyDict,
    strategies: dict,
    save_dir: str,
) -> None:
    """Setup logging and aggregate metrics from multiple runs."""
    log_filename = datetime.now().strftime(
        f"{config_dataset.dataset['type']}[%Y-%m-%d-%H-%M].log"
    )
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s | %(funcName)s"
    log_path = os.path.join(config.save_path, config.resume.split("/")[0], "log")
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, log_filename)
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format=fmt)

    logging.info(f"{config.k_shot}-shot anomaly detection using device {config.gpu}.")

    non_remain_strategies = {k: v for k, v in strategies.items() if v != "remain"}
    message = f"""
    Basic Information:
        Repetitions:        {config.aggregate_metrics}
        Backbone:           {config.backbone}
        Layers:             {config.layers_to_extract_from}
        Strategy:           {non_remain_strategies}
        Batch size:         {config.batch_size}
        Resize:             {tuple(config.resize)}
        Crop:               {tuple(config.crop_size)}
        Input shape:        {(3, *config.crop_size)}
        Feature dim:        {config.feat_dim}
        Projection dim:     {config.proj_dim}
        Feature crop ratio: {config.feat_crop_ratio}
        Nbr:                {config.nbr}
        Global memory:      {config.glo_memory_num}
        Local memory:       {config.loc_memory_num}
        K-min:              {config.k_min}
        Padding mode:       {config.padding_mode}
    """
    logging.info(message)
    print(message)

    aggregate_metrics(os.path.join(save_dir, "metrics.csv"), config.seed + 1)


def main() -> None:
    """Main entry point."""
    config = parse_args()
    # config.is_plot = True
    fix_seeds(config.seed)
    run(config)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()