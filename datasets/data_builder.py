from datasets.custom_dataset import build_custom_dataloader
from datasets.generic_dataset import build_generic_dataloader
from torch.utils.data import Dataset
import argparse
import yaml
from easydict import EasyDict
from torch.utils.data import ConcatDataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def build(cfg, training, distributed):
    if training:
        cfg.update(cfg.get("train", {}))
    else:
        cfg.update(cfg.get("test", {}))

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "visa":
        data_loader = build_generic_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True):
    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, training=True, distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, training=False, distributed=distributed)

    print("Build dataset done.")
    return train_loader, test_loader


def build_mixed_dataloader(first_config_path, sec_config_path):

    with open(first_config_path) as first_file:
        first_config = EasyDict(yaml.load(first_file, Loader=yaml.FullLoader))
    first_dataset = first_config.dataset
    batch_size = first_dataset.batch_size
    num_workers = first_dataset.workers
    first_train_loader, first_val_loader = build_dataloader(first_config.dataset, distributed=False)

    with open(sec_config_path) as sec_file:
        sec_config = EasyDict(yaml.load(sec_file, Loader=yaml.FullLoader))
    sec_train_loader, sec_val_loader = build_dataloader(sec_config.dataset, distributed=False)

    train_loader_dataset = ConcatDataset([first_train_loader.dataset, sec_train_loader.dataset])
    val_loader_dataset = ConcatDataset([first_val_loader.dataset, sec_val_loader.dataset])

    train_loader = DataLoader(train_loader_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_loader_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_loader, val_loader


if __name__ == '__main__':
    import torch
    from tool.visualization import show_batch_images
    from torch.utils.data import Subset, SubsetRandomSampler
    from warnings import filterwarnings
    filterwarnings('ignore')


    def test_one(config_path="./config_dataset.yaml"):
        parser = argparse.ArgumentParser(description="Dataset")
        parser.add_argument("--config", default=config_path)
        args = parser.parse_args()
        with open(args.config) as f:
            config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        _train_loader, _val_loader = build_dataloader(config.dataset, distributed=False)
        for data in _train_loader:
            print(data['image'])
            show_batch_images(data['image'])
            break
        for data in _val_loader:
            # print(data['image'])
            show_batch_images(data['image'])
            show_batch_images(data['mask'], need_de_normalize=False)
            break
        all_test_dataset = _val_loader.dataset
        sub_val_dataset_ = Subset(all_test_dataset, range(int(len(all_test_dataset) * 0.36)))
        print(len(sub_val_dataset_), len(all_test_dataset))
        sub_test_dataset_size = int(len(all_test_dataset) * 0.36)

        sampler_indices = torch.randperm(len(all_test_dataset))[:sub_test_dataset_size]
        sampler = SubsetRandomSampler(sampler_indices)
        sub_val_loader = DataLoader(all_test_dataset, batch_size=8, sampler=sampler)
        for data in sub_val_loader:
            # print(data['image'])
            show_batch_images(data['image'])
            break
        if config_path == "./config_visa_dataset.yaml":
            cls_dataloader = build_generic_dataloader(config.dataset, False, False, class_name='pcb1')
            for data in cls_dataloader:
                print(data['image'])
                show_batch_images(data['image'])
                show_batch_images(data['mask'], need_de_normalize=False)
                break


    def test_two():
        parser = argparse.ArgumentParser(description="Dataset")
        parser.add_argument("--config", default="./config_visa_dataset.yaml")
        args = parser.parse_args()
        with open(args.config) as f:
            config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        _train_loader_visa, _val_loader_visa = build_dataloader(config.dataset, distributed=False)
        parser_ = argparse.ArgumentParser(description="Dataset_")
        parser_.add_argument("--config", default="./config_bt_dataset.yaml")
        args_ = parser_.parse_args()
        with open(args_.config) as f_:
            config = EasyDict(yaml.load(f_, Loader=yaml.FullLoader))
        _train_loader_bt, _val_loader_bt = build_dataloader(config.dataset, distributed=False)
        _train_loader_dataset = ConcatDataset([_train_loader_visa.dataset, _train_loader_bt.dataset])
        _val_loader_dataset = ConcatDataset([_val_loader_visa.dataset, _val_loader_bt.dataset])
        _train_loader = DataLoader(_train_loader_dataset, batch_size=8, num_workers=4, shuffle=True)
        _val_loader = DataLoader(_val_loader_dataset, batch_size=8, num_workers=4, shuffle=True)

        for data in _train_loader:
            print(data['image'])
            show_batch_images(data['image'])
            show_batch_images(data['mask'], need_de_normalize=False)
            break
        for data in _val_loader:
            # print(data['image'])
            show_batch_images(data['image'])
            show_batch_images(data['mask'], need_de_normalize=False)
            break


    # test_one("./config_visa_dataset.yaml")
    test_one()
    # test_one("./config_bt_dataset.yaml")
    # test_one("./config_mdpp_dataset.yaml")
    # test_two()
