from __future__ import division

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
from PIL import Image
from torchvision import transforms
import numpy as np
import csv


VISA_CLASS_NAMES = [
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "macaroni1",
    "macaroni2",
    "capsules",
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
]

BT_CLASS_NAMES = [
    "product01",
    "product02",
    "product03",
]

DTD_CLASS_NAMES = [
    "woven_001",
    "woven_068",
    "woven_104",
    "woven_125",
    "woven_127",
    "stratified_154",
    "blotchy_099",
    "marbled_078",
    "perforated_037",
    "mesh_114",
    "fibrous_183",
    "matted_069",
]

RAD_CLASS_NAMES = [
    "bolt",
    "ribbon",
    "sponge",
    "tape",
]

CI_CLASS_NAMES = [
    "cable_1",
    "cable_2",
    "cable_3",
]


def build_generic_dataloader(cfg, training, distributed=True, class_name=''):
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

    print("building Dataset from: {}".format(cfg["meta_file"]))

    dataset = UniDataset(
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


class UniDataset(BaseDataset):
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
        super(UniDataset, self).__init__()
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
        if training:
            need_type = 'train'
        else:
            need_type = 'test'
        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            render = csv.reader(f_r, delimiter=',')
            next(render)  # header
            if class_name == '':
                for row in render:
                    if row[1] == need_type:
                        meta_dict = {'object': row[0], 'split': row[1], 'label': row[2], 'image': row[3], 'mask': row[4]}
                        self.metas.append(meta_dict)
            else:
                for row in render:
                    if row[1] == need_type and row[0] == class_name:
                        meta_dict = {'object': row[0], 'split': row[1], 'label': row[2], 'image': row[3], 'mask': row[4]}
                        self.metas.append(meta_dict)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["image"]
        if isinstance(meta["label"], int):
            label = meta["label"]
        else:
            label = 0 if meta["label"] == 'normal' else 1
        image = self.image_reader(meta["image"])
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("object", None):
            input["clsname"] = meta["object"]
        else:
            input["clsname"] = filename.split("/")[0]

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if meta.get("mask", None):
            mask = self.image_reader(meta["mask"], is_mask=True)
            mask[mask != 0] = 255
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



