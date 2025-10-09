from __future__ import division

import json
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
from datasets.generic_dataset import UniDataset


def select_training_data(cfg, k_shot, class_name):
    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    transform_fn = TrainBaseTransform(
        cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"], cfg["crop_size"]
    )

    colorjitter_fn = None
    if cfg.get("colorjitter", None):
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    print("select normal samples from: {}".format(cfg["meta_file"]))
    try:
        dataset = CustomDataset(
            image_reader,
            cfg["meta_file"],
            True,
            transform_fn=transform_fn,
            normalize_fn=normalize_fn,
            colorjitter_fn=colorjitter_fn,
            class_name=class_name
        )
    except Exception as e:
        print(f"Error occurred while creating CustomDataset: {e}")
        print("Switching to UniDataset...")

        dataset = UniDataset(
            image_reader,
            cfg["meta_file"],
            True,
            transform_fn=transform_fn,
            normalize_fn=normalize_fn,
            colorjitter_fn=colorjitter_fn,
            class_name=class_name
        )

    indices = random.sample(range(len(dataset)), k_shot)
    selected_data = [dataset[idx] for idx in indices]

    return selected_data


def build_custom_dataloader(cfg, training, distributed=True, class_name=''):
    image_reader = build_image_reader(cfg.image_reader)

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"], cfg["crop_size"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"], cfg["crop_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    print("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        image_reader,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
        class_name=class_name
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class CustomDataset(BaseDataset):
    def __init__(
            self,
            image_reader,
            meta_file,
            training,
            transform_fn,
            normalize_fn,
            colorjitter_fn=None,
            class_name=''
    ):
        super(CustomDataset, self).__init__()
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            if class_name == '':
                for line in f_r:
                    meta = json.loads(line)
                    self.metas.append(meta)
            else:
                for line in f_r:
                    meta = json.loads(line)
                    if meta['clsname'] == class_name:
                        self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if meta.get("maskname", None):
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            mask = np.zeros((image.height, image.width)).astype(np.uint8)

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        input.update({"image": image, "mask": mask})
        return input


if __name__ == '__main__':
    # import torch
    # dataset = [1, 2, 3, 4, 5, 6, 7, 8, 0, 11, 9, 10, 12, 14, 15]
    # loader = DataLoader(dataset, batch_size=2)
    # # for i, data in enumerate(loader):
    # #     print(data)
    # #     if i > 2:
    # #         break
    # #
    # # for i, data in enumerate(loader):
    # #     print(data)
    # #     if i > 2:
    # #         break
    # print(loader.dataset)
    # print(len(loader))
    # num = int(len(loader) / 2)
    # print(num)
    # from torch.utils.data import Subset
    # sub_dataset = Subset(loader.dataset, range(num))
    # sub_loader = DataLoader(sub_dataset, batch_size=2)
    # print(sub_dataset, sub_loader)
    # for data in sub_loader:
    #     print(data)
    import argparse
    from easydict import EasyDict
    import yaml

    parser = argparse.ArgumentParser(description="Dataset")
    parser.add_argument("--config", default="./config_dataset.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    cfg_dataset = config.dataset
    cfg_dataset.update(cfg_dataset.get("train", None))
    x = select_training_data(cfg_dataset, 3, 'bottle')
    for i in range(len(x)):
        print(x[i]['filename'])
    print(x)
