import os
import lib
import torch
import argparse
import warnings
import yaml
import copy
import logging
from datetime import datetime
from easydict import EasyDict
from datasets import *

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser('DyNorm')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--save_path', type=str, default='./results', help="the path to save the results")
    parser.add_argument('--backbone', type=str, default='wideresnet50', help="backbone name")
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--size', '-s', type=int, nargs=4,
                        metavar=('resize_width', 'resize_height', 'crop_width', 'crop_height'),
                        default=[],
                        help='Resize dimensions and crop size (resize_width resize_height crop_width crop_height)')
    parser.add_argument('--layers_to_extract_from', '-le', type=str, nargs='+', default=['layer2', 'layer3'],
                        help='feature extraction layer selection')
    parser.add_argument('--nbr', type=int, default=3, help='neighborhood size')
    parser.add_argument('--feat_dim', '-fd', type=int, default=1024, help='extracted feature dimensions')
    parser.add_argument('--proj_dim', '-pd', type=int, default=256, help='dimensionality reduction')
    parser.add_argument('--glo_memory_num', '-gm', type=int, default=12, help='global memory bank size')
    parser.add_argument('--loc_memory_num', '-lm', type=int, default=3, help='local memory bank size')
    parser.add_argument('--feat_crop_ratio', '-fcr', type=float, default=0.9, help='feature clipping')
    parser.add_argument('--k_min', '-km', type=float, default=0.05, help='minimum')
    parser.add_argument('--gpu', type=int, default=0, help='GPU selection')
    parser.add_argument('--k_shot', '-k', type=int, default=0, help='k-shot')
    parser.add_argument('--is_plot', '-plt', action='store_true', help='whether to draw or not')
    parser.add_argument('--aggregate_metrics', '-am', type=int, default=-1,
                        help='aggregates metrics from multiple CSV files, calculates the mean and standard deviation')
    parser.add_argument('--resume', type=str, default='exp1st', help='change directory')
    parser.add_argument('--padding_mode', '-pm', type=str, default='border', help='registration padding mode')
    parser.add_argument("--cfg", default="./datasets/config_dataset.yaml")
    return parser.parse_args()


def run(config):
    device = lib.set_torch_device([config.gpu])

    metrics = {}

    print(f'{config.k_shot}-shot anomaly detection by using device {config.gpu}.')
    print_message = 'All class: '

    all_class = ''
    with open(config.cfg) as f:
        config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    if config_dataset.dataset["type"] == 'custom':
        dataset_class_names = CLASS_NAMES
        choose_strategy = {
            'screw': 'transform',
            'metal_nut': 'transform',
            **{name: 'remain' for name in CLASS_NAMES if
               name not in ['screw', 'metal_nut']}
        }
    elif config_dataset.dataset["type"] == 'visa':
        dataset_class_names = VISA_CLASS_NAMES
        choose_strategy = {
            **{name: 'remain' for name in VISA_CLASS_NAMES}
        }
        # choose_strategy = {
        #     'pcb1': 'transform',
        #     'fryum': 'transform',
        #     'cashew': 'transform',
        #     **{name: 'translate' for name in VISA_CLASS_NAMES if
        #        name not in ['pcb1', 'fryum', 'cashew']}
        # }
    elif config_dataset.dataset["type"] == 'dtd':
        dataset_class_names = DTD_CLASS_NAMES
        choose_strategy = {
            **{name: 'remain' for name in DTD_CLASS_NAMES}
        }
    elif config_dataset.dataset["type"] == 'rad':
        dataset_class_names = RAD_CLASS_NAMES
        choose_strategy = {
            **{name: 'remain' for name in RAD_CLASS_NAMES}
        }
    elif config_dataset.dataset["type"] == 'ci':
        dataset_class_names = CI_CLASS_NAMES
        choose_strategy = {
            **{name: 'remain' for name in CI_CLASS_NAMES}
        }
    else:
        dataset_class_names = BT_CLASS_NAMES
        choose_strategy = {
            **{name: 'remain' for name in BT_CLASS_NAMES}
        }

    for c_n in dataset_class_names:
        all_class += c_n + ' '
    config_dataset.dataset["batch_size"] = config.batch_size
    config.resume = f'{config.resume}/batch_{config.batch_size}/seed_{config.seed}/' \
                    f'{config.k_shot}_shot_{config_dataset.dataset["type"]}[{config.backbone}]'
    save_dir = os.path.join(config.save_path, config.resume)
    os.makedirs(save_dir, exist_ok=True)

    print_message += f"{all_class}\n"
    print_message += f"Info: | Backbone name: {config.backbone} | "
    print_message += f"Layers: {config.layers_to_extract_from} | "
    print_message += f"Batch size: {config.batch_size} | "

    if config.size:
        config_dataset.dataset['input_size'] = config.size[:2]
        config_dataset.dataset['crop_size'] = config.size[2:4]

    config.resize = config_dataset.dataset['input_size']
    config.crop_size = config_dataset.dataset['crop_size']
    print_message += f"Scale: {config.resize} | "
    print_message += f"Cropping: {config.crop_size} | "
    print(print_message)

    cfg_dataset = config_dataset.dataset
    train_dataset = copy.deepcopy(cfg_dataset)
    cfg_dataset.update(cfg_dataset.get("test", None))
    train_dataset.update(train_dataset.get("train", None))

    for class_name in dataset_class_names:
        if config_dataset.dataset["type"] == 'custom':
            cls_dataloader = build_custom_dataloader(cfg_dataset, False, False, class_name=class_name)
        else:
            cls_dataloader = build_generic_dataloader(cfg_dataset, False, False, class_name=class_name)

        info_set = {"dataset_set": cls_dataloader, "cls_name": class_name}

        model = lib.DyNorm(config, lib.load(config.backbone), config.layers_to_extract_from,
                           info_set, device, (3, *config.crop_size), config.feat_dim, config.feat_dim,
                           3, 1, proj_dim=config.proj_dim, strategy=choose_strategy[class_name]
                           )
        if config.k_shot > 0:
            train_data = select_training_data(train_dataset, k_shot=config.k_shot, class_name=class_name)
            model.few_shot_memory(train_data)
        metrics[class_name] = model.test()
        torch.cuda.empty_cache()
    lib.save_metrics_to_csv(save_dir, metrics)

    if config.aggregate_metrics > 0:
        log_filename = datetime.now().strftime(f"{config_dataset.dataset['type']}[%Y-%m-%d-%H-%M].log")
        fmt = '%(asctime)s | %(name)s | %(levelname)s | %(message)s | %(funcName)s'
        log_path = os.path.join(config.save_path, config.resume.split('/')[0], "log")
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, log_filename)
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format=fmt)
        logging.info(f'{config.k_shot}-shot anomaly detection by using device {config.gpu}.')
        choose_strategy_print = {k: v for k, v in choose_strategy.items() if v != 'remain'}
        massage = f'''
                    Basic information:
                        Repetition count:   {config.aggregate_metrics}
                        Backbone name:      {config.backbone}
                        Layers:             {config.layers_to_extract_from}
                        Strategy:           {choose_strategy_print}
                        Batch size:         {config.batch_size}
                        Scale:              {tuple(config.resize)}
                        Cropping:           {tuple(config.crop_size)}
                        Input shape:        {(3, *config.crop_size)}
                        Feature dim:        {config.feat_dim}
                        Projection dim:     {config.proj_dim}
                        Feature crop ratio: {config.feat_crop_ratio}
                        Nbr:                {config.nbr}
                        GM num:            {config.glo_memory_num}
                        LM num:            {config.loc_memory_num}
                        K-min:              {config.k_min}
                        Padding mode:       {config.padding_mode}
                    '''
        logging.info(massage)
        print(massage)
        lib.aggregate_metrics(os.path.join(save_dir, 'metrics.csv'), config.seed + 1)


def main():
    config = parse_args()
    lib.utils.fix_seeds(config.seed)
    run(config)


if __name__ == '__main__':
    main()
