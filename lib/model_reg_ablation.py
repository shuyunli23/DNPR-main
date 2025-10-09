import tqdm
import time
import math
from torch import optim
from lib.utils import os, torch, normalize_data, select_reference_image_with_orb
from lib.common import AffineTrans
from lib.model import DyNorm
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score


class DyNormRegAblate(DyNorm):

    def __init__(
            self,
            args,
            backbone,
            layers_to_extract_from,
            dataloader,
            device,
            input_shape,
            pretrain_embed_dimension=1024,
            target_embed_dimension=1024,
            patch_size=3,
            patch_stride=1,
            top_k=9,
            proj_dim=256,
            strategy='transform',
            first_batch_lower_limit=4,
            transform_choose='pmgr'
    ):
        super(DyNormRegAblate, self).__init__(args, backbone, layers_to_extract_from, dataloader, device, input_shape,
                                              pretrain_embed_dimension, target_embed_dimension, patch_size,
                                              patch_stride, top_k, proj_dim, strategy, first_batch_lower_limit)
        self.mse, self.mae, self.mi, self.ssim = 0., 0., 0., 0.
        self.transform_choose = transform_choose

    def registration(self, ref_img, align_img):

        matrix, model = self.batch_img_registration(
            self.re_size(ref_img, 128), self.re_size(align_img, 128)
        )
        # matrix, model = self.batch_img_registration(ref_img, align_img)
        if self.reg_msk is None:
            model.img_shape = ref_img.size(2)
            self.reg_msk = model.generate_circular_mask().to(self.device)
        registered = model.affine_trans(matrix, align_img, padding_mode=self.args.padding_mode, need_blank_mask=False)

        return registered, model, matrix

    def batch_img_registration(self, ref_img, align_img, lr_theta=0.1, lr_trans=0.005,
                               num_epochs=100, error_tolerance_factor=0.6):
        b, _, h, _ = align_img.shape
        num_epochs = max(int((b / 16) * num_epochs), 30)
        if self.transform_choose == 'affine':
            model = AffineTrans(batch_size=b, img_shape=h, is_affine=True).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr_theta)
            start_matrix = model.get_affine_matrix()
            registered = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs)
            matrix = model.get_affine_matrix()
            _, blank_msk = model.affine_trans(matrix, align_img, padding_mode='zeros')

            error_start = torch.mean((align_img * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                                     dim=(1, 2, 3))
            error = torch.mean((registered * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                               dim=(1, 2, 3))

            condition2 = error > (error_start * error_tolerance_factor)
            matrix = matrix.clone()
            matrix[condition2] = start_matrix[condition2]
        else:
            model = AffineTrans(batch_size=b, img_shape=h).to(self.device)
            if self.transform_choose != 'pmgr':
                lr_trans = lr_theta

            # Prepare optimizer parameters
            optimizer_params = [{'params': model.translation, 'lr': lr_trans}]
            if self.use_rotation:
                optimizer_params.append({'params': [model.theta], 'lr': lr_theta})
            else:
                model.fixed_parameters(fixed_rotation=True)

            optimizer = optim.Adam(optimizer_params)
            start_matrix = model.get_affine_matrix()
            if self.transform_choose == 'pmgr':
                if self.use_rotation:
                    # First optimization phase
                    registered_1st = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs)
                    matrix = model.get_affine_matrix()
                    _, blank_msk_1st = model.affine_trans(matrix, align_img, padding_mode='zeros')

                    # Adjust theta and perform second optimization phase
                    model.theta.data.copy_(model.theta.data + math.pi)
                    registered_2nd = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs)
                    matrix_2nd = model.get_affine_matrix()
                    _, blank_msk_2nd = model.affine_trans(matrix_2nd, align_img, padding_mode='zeros')

                    # Calculate merged mask and errors
                    merged_mask = (~torch.logical_or(blank_msk_1st, blank_msk_2nd)).float()
                    error_1st = torch.mean((registered_1st * merged_mask - ref_img * merged_mask) ** 2,
                                           dim=(1, 2, 3))
                    error_2nd = torch.mean((registered_2nd * merged_mask - ref_img * merged_mask) ** 2,
                                           dim=(1, 2, 3))

                    # Determine conditions for matrix update
                    error_start = torch.mean((align_img * merged_mask - ref_img * merged_mask) ** 2, dim=(1, 2, 3))
                    condition1 = error_1st > error_2nd
                    condition2 = (error_1st > (error_start * error_tolerance_factor)) & \
                                 (error_2nd > (error_start * error_tolerance_factor))
                    matrix[condition1] = matrix_2nd[condition1]
                else:
                    # Translation-only optimization
                    registered = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs // 4)
                    matrix = model.get_affine_matrix()
                    _, blank_msk = model.affine_trans(matrix, align_img, padding_mode='zeros')

                    error_start = torch.mean((align_img * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                                             dim=(1, 2, 3))
                    error = torch.mean((registered * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                                       dim=(1, 2, 3))

                    condition2 = error > (error_start * error_tolerance_factor)
            else:
                if self.use_rotation:
                    registered = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs)
                else:
                    registered = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs // 4)
                matrix = model.get_affine_matrix()
                _, blank_msk = model.affine_trans(matrix, align_img, padding_mode='zeros')

                error_start = torch.mean((align_img * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                                         dim=(1, 2, 3))
                error = torch.mean((registered * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                                   dim=(1, 2, 3))

                condition2 = error > (error_start * error_tolerance_factor)

            # Update matrix based on conditions
            matrix[condition2] = start_matrix[condition2]

        return matrix, model

    def matrix_opt(self, ref_img, align_img, model, optimizer, num_epochs):
        registered_img, _ = model(align_img)

        for epoch in range(num_epochs):
            if self.transform_choose == 'pmgr':
                mask = model.mask.to(self.device)
                loss = torch.mean((ref_img * mask - registered_img * mask) ** 2)
            else:
                loss = torch.mean((ref_img - registered_img) ** 2)

            # loss = torch.mean((ref_img * model.mask - registered_img * model.mask) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            registered_img, _ = model(align_img)

        return registered_img

    def predict(self):
        all_scores_image = []
        gt_labels = []
        all_scores_pixel = []
        labels = []
        images = []
        regs = []
        reg_model = None

        below_first_batch_lower_limit = True if self.first_batch_lower_limit > self.args.batch_size else False
        data_len = len(self.dataloader['dataset_set'])
        backup_data = None

        first_img = self.knowledge['images'] if self.k_shot != 0 else None
        start_time = time.time()
        is_start = True
        for idx, data in tqdm.tqdm(enumerate(self.dataloader['dataset_set'], 0), desc="Predicting...", leave=False,
                                   total=data_len):
            if below_first_batch_lower_limit and is_start:
                backup_data = {k: data[k] for k in ['image', 'label', 'mask']}
                continue

            if below_first_batch_lower_limit:
                for key in ['image', 'label', 'mask']:
                    backup_data[key] = torch.cat([backup_data[key], data[key]], dim=0)
                if self.first_batch_lower_limit > len(backup_data['image']):
                    continue
                else:
                    input_images = backup_data['image']
                    img_label = backup_data['label']
                    img_mask = backup_data['mask']
                    below_first_batch_lower_limit = False
            else:
                input_images = data['image']
                img_label = data['label']
                img_mask = data['mask']

            # GD-IRA
            if self.strategy == 'remain':
                reg_images = input_images.to(self.device)
                matrix, reg_model = None, None
            else:
                if is_start and self.k_shot == 0:
                    selected_index = select_reference_image_with_orb(input_images)
                    first_img = input_images[selected_index][None].to(self.device)
                registered, reg_model, matrix = self.registration(first_img, input_images.to(self.device))
                reg_images = registered.detach()

            scores_image, scores_pixel = self.batch_predict(reg_images, matrix, reg_model)
            is_start = False
            for scores in scores_pixel:
                all_scores_pixel.append(torch.from_numpy(scores).unsqueeze(0))
            labels.append(img_label)
            all_scores_image.append(scores_image.view(-1))
            gt_labels.append(img_mask)
            images.append(input_images)
            regs.append(reg_images)

        end_time = time.time()
        all_scores_pixel = torch.cat(all_scores_pixel, dim=0)
        labels = torch.cat(labels, dim=0)
        all_scores_image = torch.cat(all_scores_image, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        images = torch.cat(images, dim=0)

        regs = torch.cat(regs, dim=0)
        reg_model.img_shape = first_img.size(-1)
        msk = reg_model.generate_circular_mask().to(self.device)
        self.mse = torch.mean((regs * msk - first_img * msk) ** 2).item()
        self.mae = torch.mean(torch.abs(regs * msk - first_img * msk)).item()
        self.mi = compute_mi(first_img * msk, regs * msk)
        self.ssim = compute_ssim(first_img * msk, regs * msk)

        return normalize_data(all_scores_image), normalize_data(
            all_scores_pixel), labels, gt_labels, end_time - start_time, images, regs.cpu()


def compute_ssim(ref, regs):
    ref_np = ref.cpu().detach().numpy().squeeze()

    ssim_values = []
    for i in range(regs.shape[0]):
        regs_np = regs[i].cpu().detach().numpy().squeeze()
        ssim_value = ssim(regs_np, ref_np, data_range=ref_np.max() - ref_np.min(), channel_axis=0)
        ssim_values.append(ssim_value)

    average_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    return average_ssim


def compute_mi(ref, regs):
    ref_np = ref.cpu().detach().numpy().flatten()
    mi_values = []
    for i in range(regs.shape[0]):
        regs_np = regs[i].cpu().detach().numpy().flatten()
        mi_value = mutual_info_score(regs_np, ref_np)
        mi_values.append(mi_value)

    average_mi = sum(mi_values) / len(mi_values) if mi_values else 0.0
    return average_mi


if __name__ == '__main__':
    import torch.profiler
    import lib
    import yaml
    from easydict import EasyDict
    import argparse
    import warnings
    from datasets import *
    import copy
    import logging
    import numpy as np

    warnings.filterwarnings('ignore')
    path_set = "./"


    def test_1st(conf, seed=0):
        logging.info(f"\nSeed-{seed}:")
        lib.utils.fix_seeds(seed)
        _device = lib.set_torch_device([0])
        metrics = {}
        mse_col, mae_col, mi_col, ssim_col = {}, {}, {}, {}
        for l in conf.choose:
            if l == 'mvtec':
                with open(f"{path_set}datasets/config_dataset.yaml") as f:
                    config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
                dataset_class_names = CLASS_NAMES
                choose_strategy = {
                    'screw': 'transform',
                    'metal_nut': 'transform',
                    **{name: 'remain' for name in CLASS_NAMES if
                       name not in ['screw', 'metal_nut']}
                }
            elif l == 'visa':
                with open(f"{path_set}datasets/config_visa_dataset.yaml") as f:
                    config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
                dataset_class_names = VISA_CLASS_NAMES
                choose_strategy = {
                    **{name: 'remain' for name in VISA_CLASS_NAMES}
                }
            elif l == 'ci':
                with open(f"{path_set}datasets/config_ci_dataset.yaml") as f:
                    config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
                dataset_class_names = CI_CLASS_NAMES
                choose_strategy = {
                    **{name: 'remain' for name in CI_CLASS_NAMES}
                }
            else:
                with open(f"{path_set}datasets/config_bt_dataset.yaml") as f:
                    config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
                dataset_class_names = BT_CLASS_NAMES
                choose_strategy = {
                    **{name: 'remain' for name in BT_CLASS_NAMES}
                }

            cfg_dataset = config_dataset.dataset
            train_dataset = copy.deepcopy(cfg_dataset)
            cfg_dataset.update(cfg_dataset.get("test", None))
            train_dataset.update(train_dataset.get("train", None))
            for class_name in dataset_class_names:
                if choose_strategy[class_name] != 'transform':
                    continue
                # class_name = 'screw'
                print(class_name)
                if l == 'mvtec':
                    cls_dataloader = build_custom_dataloader(cfg_dataset, False, False, class_name=class_name)
                else:
                    cls_dataloader = build_generic_dataloader(cfg_dataset, False, False, class_name=class_name)

                info_set = {"dataset_set": cls_dataloader, "cls_name": class_name}
                model = DyNormRegAblate(conf,
                                        lib.load(conf.backbone),
                                        conf.layers_to_extract_from,
                                        info_set,
                                        _device,
                                        (3, conf.crop_size, conf.crop_size),
                                        conf.feat_dim, conf.feat_dim, 3, 1, strategy=choose_strategy[class_name],
                                        proj_dim=256, transform_choose=conf.transform_choose
                                        )
                if conf.k_shot > 0:
                    train_data = select_training_data(train_dataset, k_shot=conf.k_shot, class_name=class_name)
                    model.few_shot_memory(train_data)

                metrics[class_name] = model.test()
                mse_col[class_name] = model.mse
                mae_col[class_name] = model.mae
                mi_col[class_name] = model.mi
                ssim_col[class_name] = model.ssim
                torch.cuda.empty_cache()
        total_img_auroc = []
        total_pixel_auroc = []
        total_img_ap = []
        total_pixel_ap = []
        total_img_f1 = []
        total_pixel_f1 = []
        total_pro = []
        total_time = []
        total_mse = []
        total_mae = []
        total_mi = []
        total_ssim = []
        log_rows = []
        header = ['Cls-name', 'I-AUROC', 'P-AUROC', 'I-AP', 'P-AP', 'I-F1', 'P-F1', 'PRO', 'Speed', 'MSE', 'MAE',
                  'MI', 'SSIM']
        log_rows.append(
            '{:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}'.format(
                *header))
        for class_name, res in metrics.items():
            row = [
                class_name,
                round(res['img_auc'] * 100, 2),
                round(res['pixel_auc'] * 100, 2),
                round(res['img_ap'] * 100, 2),
                round(res['pixel_ap'] * 100, 2),
                round(res['img_f1'] * 100, 2),
                round(res['pixel_f1'] * 100, 2),
                round(res['pro'] * 100, 2),
                round(res['test_time'], 2),
                round(mse_col[class_name], 2),
                round(mae_col[class_name], 2),
                round(mi_col[class_name], 2),
                round(ssim_col[class_name], 2)
            ]

            log_rows.append(
                '{:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}'.format(
                    *row))

            total_img_auroc.append(res['img_auc'])
            total_pixel_auroc.append(res['pixel_auc'])
            total_img_ap.append(res['img_ap'])
            total_pixel_ap.append(res['pixel_ap'])
            total_img_f1.append(res['img_f1'])
            total_pixel_f1.append(res['pixel_f1'])
            total_pro.append(res['pro'])
            total_time.append(res['test_time'])
            total_mse.append(mse_col[class_name])
            total_mae.append(mae_col[class_name])
            total_mi.append(mi_col[class_name])
            total_ssim.append(ssim_col[class_name])

        average_row = [
            'Average',
            round(np.mean(total_img_auroc) * 100, 2),
            round(np.mean(total_pixel_auroc) * 100, 2),
            round(np.mean(total_img_ap) * 100, 2),
            round(np.mean(total_pixel_ap) * 100, 2),
            round(np.mean(total_img_f1) * 100, 2),
            round(np.mean(total_pixel_f1) * 100, 2),
            round(np.mean(total_pro) * 100, 2),
            round(np.mean(total_time), 2),
            round(np.mean(total_mse), 2),
            round(np.mean(total_mae), 2),
            round(np.mean(total_mi), 2),
            round(np.mean(total_ssim), 2)
        ]

        log_rows.append(
            '{:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6} {:<6}'.format(
                *average_row))
        logging.info("\n" + "\n".join(log_rows))
        return average_row[1:]


    def test_2nd(choose=None, times=5, info="pmgr"):
        if choose is None:
            choose = ["mvtec"]
        conf = argparse.Namespace(save_path=f'{path_set}results')
        conf.resume = f'test_reg_ablate'
        conf.layers_to_extract_from = ['layer2', 'layer3']
        conf.crop_size = 336
        conf.batch_size = 16
        conf.k_min = 0.05
        conf.glo_memory_num = 12
        conf.feat_dim = 1024
        conf.is_plot = True
        conf.padding_mode = 'border'
        conf.k_shot = 0
        conf.loc_memory_num = 3
        conf.feat_crop_ratio = 0.9
        conf.nbr = 9
        conf.backbone = 'wideresnet50'

        conf.choose = choose
        conf.transform_choose = info
        save_dir = os.path.join(conf.save_path, conf.resume)
        os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(
            filename=f'{save_dir}/metrics-{info}.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        _ = test_1st(conf, seed=3)
        exit()
        avg_col = []
        for sd in range(times):
            print(f'Seed-{sd}:')
            avg = test_1st(conf, seed=sd)
            avg_col.append(avg)
        avg_np = np.array(avg_col)
        logging.info("all results:\n%s", avg_np)
        logging.info("all mean:\n%s", np.mean(avg_np, axis=0))
        logging.info("all std:\n%s", np.std(avg_np, axis=0))
        print(f'Log path: {save_dir}/metrics-{info}.log')


    # test_2nd(info="0", times=2)
    # test_2nd(choose="visa", info="1")
    # test_2nd(choose=["mvtec", "visa"], info="affine")
    # test_2nd(info="affine")
    test_2nd(info="rigid")
    # test_2nd(info="pmgr")
