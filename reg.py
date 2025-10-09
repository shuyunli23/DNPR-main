import torch
from torch import nn
import math
from torch import optim



class AffineTrans(nn.Module):
    def __init__(self, batch_size, img_shape=224, mask_method='circular', padding_mode='border', device='cpu'):
        super(AffineTrans, self).__init__()
        self.device = device
        self.padding_mode = padding_mode

        # Define the learnable theta and translation
        self.theta = nn.Parameter(torch.zeros(batch_size, 1, dtype=torch.float32))
        self.translation = nn.Parameter(torch.zeros((batch_size, 2), dtype=torch.float32))

        # Make mask
        self.img_shape = img_shape
        if mask_method == 'circular':
            self.mask = self.generate_circular_mask()
        else:
            self.mask = self.generate_four_corner_mask()

    def generate_circular_mask(self):
        mask = torch.ones(1, 1, self.img_shape, self.img_shape, dtype=torch.float32)
        center_x = int(self.img_shape / 2)
        center_y = center_x
        radius = center_y * 1.2
        y, x = torch.meshgrid(torch.arange(self.img_shape), torch.arange(self.img_shape), indexing='ij')
        mask[:, :, (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2] = 0
        return mask.to(self.device)

    def generate_four_corner_mask(self):
        mask = torch.ones(1, 1, self.img_shape, self.img_shape, dtype=torch.float32)
        side_length = int(self.img_shape / 5)

        # Mask 4 corners
        mask[:, :, :side_length, :side_length] = 0.0  # upper left
        mask[:, :, :side_length, -side_length:] = 0.0  # upper right
        mask[:, :, -side_length:, :side_length] = 0.0  # lower left
        mask[:, :, -side_length:, -side_length:] = 0.0  # lower right

        return mask.to(self.device)

    def get_affine_matrix(self):
        cos_theta, sin_theta = torch.cos(self.theta), torch.sin(self.theta)
        t_x, t_y = self.translation[:, 0].unsqueeze(1), self.translation[:, 1].unsqueeze(1)
        row1 = torch.stack([cos_theta, -1 * sin_theta, t_x], dim=1)
        row2 = torch.stack([sin_theta, cos_theta, t_y], dim=1)
        matrix = torch.stack([row1, row2], dim=1).squeeze(3)

        return matrix

    @staticmethod
    def invert_affine_matrices(matrix):
        # N = matrix.size(0)
        A = matrix[:, :, :2]  # Extract linear part (N, 2, 2)
        t = matrix[:, :, 2]  # Extract the translation part (N, 2)

        # Calculate the inverse of A
        A_inv = torch.inverse(A)  # (N, 2, 2)

        # Calculate the new translation part
        t_inv = -torch.bmm(A_inv, t.unsqueeze(2)).squeeze(2)  # (N, 2)

        # Assemble inverse affine matrix
        matrix_inv = torch.cat([A_inv, t_inv.unsqueeze(2)], dim=2)  # (N, 2, 3)

        return matrix_inv

    @staticmethod
    def affine_trans(matrix, images, padding_mode='border'):
        grid = nn.functional.affine_grid(matrix, images.size())
        images_transformed = nn.functional.grid_sample(images, grid, padding_mode=padding_mode)
        blank_mask = torch.all(images_transformed == 0, dim=1)
        return images_transformed, blank_mask.unsqueeze(1)

    def forward(self, x):
        # Transform the input image using the current affine transformation matrix
        matrix = self.get_affine_matrix()
        x_transformed, blank_mask = self.affine_trans(matrix, x, padding_mode=self.padding_mode)
        return x_transformed, blank_mask


class FastReg:

    def __init__(
            self,
            batch_size,
            input_size,
            padding_mode,
            device,
    ):
        self.device = device
        self.padding_mode = padding_mode
        reg_model = AffineTrans(batch_size, input_size, device=self.device)
        matrix = reg_model.get_affine_matrix()
        self.reg_model = reg_model.to(self.device)
        self.matrix = matrix.to(self.device)

    def fast_registration(self, ref_img, align_img):
        matrix, model = self.batch_img_registration(ref_img, align_img)
        registered, _ = model.affine_trans(matrix, align_img, padding_mode=self.padding_mode)

        return registered, model, matrix

    def batch_img_registration(self, ref_img, align_img, lr_theta_1st=0.1, lr_theta_2nd=0.1, lr_trans=0.005,
                    first_num_epochs=100, sec_num_epochs=100, error_tolerance_factor=0.68):
        b, _, h, _ = align_img.shape
        model = AffineTrans(batch_size=b, img_shape=h, device=self.device).to(self.device)
        optimizer = optim.Adam([
            {'params': [model.theta], 'lr': lr_theta_1st},
            {'params': model.translation, 'lr': lr_trans}
        ])
        start_matrix = model.get_affine_matrix()
        registered_1st = self.matrix_opt(ref_img, align_img, model, optimizer, first_num_epochs)
        matrix = model.get_affine_matrix()
        _, blank_msk_1st = model.affine_trans(matrix, align_img, padding_mode='zeros')

        model.theta.data.copy_(model.theta.data + math.pi)
        optimizer.param_groups[0]['lr'] = lr_theta_2nd
        registered_2nd = self.matrix_opt(ref_img, align_img, model, optimizer, sec_num_epochs)
        matrix_2nd = model.get_affine_matrix()
        _, blank_msk_2nd = model.affine_trans(matrix_2nd, align_img, padding_mode='zeros')
        merged_mask = (~torch.logical_or(blank_msk_1st, blank_msk_2nd)).float()
        error_1st = torch.mean((registered_1st * merged_mask - ref_img * merged_mask) ** 2,
                               dim=(1, 2, 3))
        error_2nd = torch.mean((registered_2nd * merged_mask - ref_img * merged_mask) ** 2,
                               dim=(1, 2, 3))
        error_start = torch.mean((align_img * merged_mask - ref_img * merged_mask) ** 2, dim=(1, 2, 3))
        condition1 = error_1st > error_2nd
        condition2 = (error_1st > (error_start * error_tolerance_factor)) & \
                     (error_2nd > (error_start * error_tolerance_factor))
        matrix[condition1] = matrix_2nd[condition1]
        # matrix[condition2] = start_matrix[condition2]

        return matrix, model
    # def batch_img_registration(self, ref_img, align_img, lr_theta_1st=0.1, lr_theta_2nd=0.1, lr_trans=0.01,
    #                 first_num_epochs=60, sec_num_epochs=60, error_tolerance_factor=0.68):
    #     b, _, h, _ = align_img.shape
    #     model = AffineTrans(batch_size=b, img_shape=h, device=self.device).to(self.device)
    #     optimizer = optim.Adam([
    #         {'params': [model.theta], 'lr': lr_theta_1st},
    #         {'params': model.translation, 'lr': lr_trans}
    #     ])
    #     start_matrix = model.get_affine_matrix()
    #     registered_1st = self.matrix_opt(ref_img, align_img, model, optimizer, first_num_epochs)
    #     matrix = model.get_affine_matrix()
    #     _, blank_msk = model.affine_trans(matrix, align_img, padding_mode='zeros')
    #     error_1st = torch.mean((registered_1st * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
    #                            dim=(1, 2, 3))
    #
    #     model.theta.data.copy_(model.theta.data + math.pi)
    #     optimizer.param_groups[0]['lr'] = lr_theta_2nd
    #     registered_2nd = self.matrix_opt(ref_img, align_img, model, optimizer, sec_num_epochs)
    #     matrix_2nd = model.get_affine_matrix()
    #     _, blank_msk = model.affine_trans(matrix_2nd, align_img, padding_mode='zeros')
    #     error_2nd = torch.mean((registered_2nd * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
    #                            dim=(1, 2, 3))
    #     error_start = torch.mean((align_img * (~blank_msk) - ref_img * (~blank_msk)) ** 2, dim=(1, 2, 3))
    #     condition1 = error_1st > error_2nd
    #     condition2 = (error_1st > (error_start * error_tolerance_factor)) & \
    #                  (error_2nd > (error_start * error_tolerance_factor))
    #     matrix[condition1] = matrix_2nd[condition1]
    #     matrix[condition2] = start_matrix[condition2]
    #
    #     return matrix, model

    # @staticmethod
    # def matrix_opt(ref_img, align_img, model, optimizer, num_epochs):
    #     registered_img, _ = model(align_img)
    #
    #     for epoch in range(num_epochs):
    #         loss = torch.mean((ref_img - registered_img) ** 2)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         registered_img, _ = model(align_img)
    #
    #     return registered_img
    @staticmethod
    def matrix_opt(ref_img, align_img, model, optimizer, num_epochs):
        registered_img, _ = model(align_img)

        for epoch in range(num_epochs):
            loss = torch.mean((ref_img * model.mask - registered_img * model.mask) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            registered_img, _ = model(align_img)
            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(f'Epoch {epoch + 1}/{num_epochs}, Learning Rate for group {i}: {param_group["lr"]}')

        return registered_img


if __name__ == '__main__':
    import lib
    import yaml
    import tqdm
    import time
    import os
    from easydict import EasyDict
    import warnings
    from datasets import *
    import copy

    warnings.filterwarnings('ignore')


    def test_one():
        lib.utils.fix_seeds(4)
        _device = lib.set_torch_device([2])

        choose = "mvtec"
        metrics = {}
        if choose == 'mvtec':
            with open("./datasets/config_dataset.yaml") as f:
                config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            dataset_class_names = CLASS_NAMES
            resume = 'mvtec'
        elif choose == 'visa':
            with open("./datasets/config_visa_dataset.yaml") as f:
                config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            dataset_class_names = VISA_CLASS_NAMES
            resume = 'visa'
        else:
            with open("./datasets/config_bt_dataset.yaml") as f:
                config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            dataset_class_names = BT_CLASS_NAMES
            resume = 'bt'
        cfg_dataset = config_dataset.dataset
        train_dataset = copy.deepcopy(cfg_dataset)
        cfg_dataset.update(cfg_dataset.get("test", None))
        train_dataset.update(train_dataset.get("train", None))
        print('Separate test...')
        for class_name in dataset_class_names:
            class_name = 'metal_nut'
            print(class_name)
            if choose == 'mvtec':
                cls_dataloader = build_custom_dataloader(cfg_dataset, False, False, class_name=class_name)
            elif choose == 'visa':
                cls_dataloader = build_visa_dataloader(cfg_dataset, False, False, class_name=class_name)
            else:
                cls_dataloader = build_bt_dataloader(cfg_dataset, False, False, class_name=class_name)
            model = FastReg(16, 224, 'reflection', _device)
            data_len = len(cls_dataloader)
            backup_data = None
            first_img = None
            start_time = time.time()
            col_img = []
            col_reg_img = []

            def select_reference_image(images, use_mean=True):
                # Compute the representative image (mean or median) across the batch
                if use_mean:
                    representative_image = images.mean(dim=0)  # Compute the mean image
                else:
                    representative_image = torch.median(images, dim=0).values  # Compute the median image

                # Calculate the Mean Squared Error (MSE) between each image and the representative image
                mse = torch.mean((images - representative_image.unsqueeze(0)) ** 2,
                                 dim=(1, 2, 3))  # Calculate MSE for each image

                # Select the image with the smallest MSE
                min_mse_index = torch.argmin(mse)  # Find the index of the minimum MSE
                return min_mse_index

            for idx, data in tqdm.tqdm(enumerate(cls_dataloader, 0), desc="Register...", leave=False, total=data_len):
                if (idx + 1) == (data_len - 1):
                    backup_data = data
                    continue
                if (idx + 1) != data_len and data_len > 1:
                    input_images = data['image'].to(_device)
                    img_mask = data['mask'].to(_device)
                else:
                    input_images = torch.cat([backup_data['image'], data['image']], dim=0).to(_device)
                    img_mask = torch.cat([backup_data['mask'], data['mask']], dim=0).to(_device)
                if idx == 0:
                    most_similar_index = select_reference_image(input_images, False)
                    # most_similar_index = 16
                    indices = torch.arange(input_images.shape[0], device=_device)

                    first_img = input_images[most_similar_index][None]
                    registered, reg_model, matrix = model.fast_registration(first_img, input_images[
                        indices != most_similar_index])
                    registered, _ = reg_model.affine_trans(matrix, input_images[
                        indices != most_similar_index], padding_mode='zeros')
                    registered_images = torch.cat([first_img, registered.detach()], dim=0)

                else:
                    registered, reg_model, matrix = model.fast_registration(first_img, input_images)
                    registered, _ = reg_model.affine_trans(matrix, input_images, padding_mode='zeros')
                    registered_images = registered.detach()
                col_img.append(input_images.cpu())
                col_reg_img.append(registered_images.cpu())
            all_registered = torch.cat(col_reg_img, dim=0)
            all_images = torch.cat(col_img, dim=0)
            from tool.visualization import show_batch_images
            show_batch_images(all_images)
            show_batch_images(all_registered)
            break
            # save_dir = os.path.join('reg_res', resume)
            # os.makedirs(save_dir, exist_ok=True)
            # lib.save_metrics_to_csv(save_dir, metrics)


    test_one()
