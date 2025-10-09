import torch
from torch import nn
import math
from torch import optim
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def normalize_image(image):
    """Normalize the image to the range [0, 255] and convert to uint8."""
    image_min = image.min()
    image_max = image.max()

    # Scale to [0, 255]
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min) * 255.0
    else:
        image = np.zeros_like(image)  # If all values are the same, return a zero array.

    return image.astype(np.uint8)


class AffineTrans(nn.Module):
    def __init__(self, batch_size, img_shape=224, mask_method='circular', padding_mode='border',
                 device='cpu', use_rotation=True, use_translation=True):
        super(AffineTrans, self).__init__()
        self.device = device
        self.padding_mode = padding_mode
        self.use_rotation = use_rotation

        # Define the theta and translation
        if use_rotation:
            self.theta = nn.Parameter(torch.zeros(batch_size, 1, dtype=torch.float32))
        else:
            self.theta = torch.zeros(batch_size, 1, dtype=torch.float32, requires_grad=False)
        if use_translation:
            self.translation = nn.Parameter(torch.zeros((batch_size, 2), dtype=torch.float32))
        else:
            self.translation = torch.zeros((batch_size, 2), dtype=torch.float32, requires_grad=False)

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
        if self.use_rotation:
            cos_theta, sin_theta = torch.cos(self.theta), torch.sin(self.theta)
        else:
            cos_theta, sin_theta = torch.ones_like(self.theta), torch.zeros_like(self.theta)

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
    def affine_trans(matrix, images, padding_mode='border', need_blank_mask=True):
        grid = nn.functional.affine_grid(matrix, images.size())
        images_transformed = nn.functional.grid_sample(images, grid, padding_mode=padding_mode)

        if need_blank_mask:
            blank_mask = torch.all(images_transformed == 0, dim=1)
            return images_transformed, blank_mask.unsqueeze(1)
        else:
            return images_transformed

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
            use_rotation=True
    ):
        self.device = device
        self.padding_mode = padding_mode
        self.use_rotation = use_rotation
        reg_model = AffineTrans(batch_size, input_size, device=self.device)
        matrix = reg_model.get_affine_matrix()
        self.reg_model = reg_model.to(self.device)
        self.matrix = matrix.to(self.device)

    def fast_registration(self, ref_img, align_img):
        matrix, model = self.batch_img_registration(ref_img, align_img)
        registered, _ = model.affine_trans(matrix, align_img, padding_mode=self.padding_mode)

        return registered, model, matrix

    def batch_img_registration(self, ref_img, align_img, lr_theta=0.1, lr_trans=0.005,
                    num_epochs=15, error_tolerance_factor=0.68):
        b, _, h, _ = align_img.shape
        model = AffineTrans(batch_size=b, img_shape=h, device=self.device).to(self.device)

        # Prepare optimizer parameters
        optimizer_params = [{'params': model.translation, 'lr': lr_trans}]
        if self.use_rotation:
            optimizer_params.append({'params': [model.theta], 'lr': lr_theta})

        optimizer = optim.Adam(optimizer_params)
        start_matrix = model.get_affine_matrix()

        if self.use_rotation:
            # First optimization phase
            registered_1st = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs)
            matrix = model.get_affine_matrix()
            _, blank_msk_1st = model.affine_trans(matrix, align_img, padding_mode='zeros')

            # Adjust theta and perform second optimization phase
            # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
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
            registered = self.matrix_opt(ref_img, align_img, model, optimizer, num_epochs // 2)
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

    @staticmethod
    def matrix_opt(ref_img, align_img, model, optimizer, num_epochs):
        registered_img, _ = model(align_img)

        for epoch in range(num_epochs):
            loss = torch.mean((ref_img * model.mask - registered_img * model.mask) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            registered_img, _ = model(align_img)

        return registered_img


class CustomReg(nn.Module):
    def __init__(self, batch_size, img_shape=224, padding_mode='border', device='cpu'):
        super(CustomReg, self).__init__()
        self.device = device
        # Define the batch_size of 2x3 affine transformation matrices
        self.matrix = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] for _ in range(batch_size)],
            dtype=torch.float32
        )

    def forward(self, x):
        # Transform the input image using the current affine transformation matrix
        # x.size() should be (batch_size, channels, height, width)

        # Use the current affine transformation matrix for the batch
        # We need to reshape self.matrix to (batch_size, 2, 3) to match the input shape for affine_grid
        grid = nn.functional.affine_grid(self.matrix, x.size())

        # Sample the input image using the generated grid
        x_transformed = nn.functional.grid_sample(x, grid)

        return x_transformed

    # def register_with_sift(self, ref_img, align_img):
    #
    #     # Convert tensors to NumPy arrays
    #     ref_img_cpu = ref_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #     align_imgs_cpu = align_img.permute(0, 2, 3, 1).cpu().numpy()
    #
    #     # Normalize ref_img
    #     ref_img_cpu = normalize_image(ref_img_cpu)
    #     # ref_img_gray = cv2.cvtColor(ref_img_cpu, cv2.COLOR_RGB2GRAY)
    #     ref_img_gray = ref_img_cpu
    #
    #     sift = cv2.SIFT_create()
    #     kp1, des1 = sift.detectAndCompute(ref_img_gray, None)
    #
    #     results = []
    #     for align_img in align_imgs_cpu:
    #         # Normalize align_img
    #         align_img = normalize_image(align_img)
    #         # align_img_gray = cv2.cvtColor(align_img, cv2.COLOR_RGB2GRAY)
    #         align_img_gray = align_img
    #
    #         kp2, des2 = sift.detectAndCompute(align_img_gray, None)
    #
    #         bf = cv2.BFMatcher(crossCheck=True)
    #         matches = bf.match(des1, des2)
    #
    #         # Sort the matches by distance
    #         matches = sorted(matches, key=lambda x: x.distance)
    #
    #         # Screening for good matches
    #         good_matches = matches[:int(len(matches) * 0.75)]
    #
    #         if len(good_matches) >= 4:
    #             src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #             dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #             affine_matrix = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)[0]
    #             if affine_matrix is not None:
    #                 affine_matrix = torch.from_numpy(affine_matrix).float()
    #                 print('hh')
    #             else:
    #                 affine_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    #         else:
    #             affine_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    #         results.append(affine_matrix)
    #     self.matrix = torch.stack(results, dim=0).to(self.device)
    def register_with_sift(self, ref_img, align_img):

        # Convert tensors to NumPy arrays
        ref_img_cpu = ref_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        align_imgs_cpu = align_img.permute(0, 2, 3, 1).cpu().numpy()

        # Normalize ref_img
        ref_img_cpu = normalize_image(ref_img_cpu)
        # ref_img_gray = cv2.cvtColor(ref_img_cpu, cv2.COLOR_RGB2GRAY)
        ref_img_gray = ref_img_cpu

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ref_img_gray, None)

        results = []
        for align_img in align_imgs_cpu:
            # Normalize align_img
            align_img = normalize_image(align_img)
            # align_img_gray = cv2.cvtColor(align_img, cv2.COLOR_RGB2GRAY)
            align_img_gray = align_img

            kp2, des2 = sift.detectAndCompute(align_img_gray, None)

            bf = cv2.BFMatcher(crossCheck=True)
            matches = bf.match(des1, des2)

            # Sort the matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Screening for good matches
            good_matches = matches[:int(len(matches) * 0.75)]

            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                affine_matrix = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)[0]
                if affine_matrix is not None:
                    cols, rows, _ = align_img.shape
                    reg_img = cv2.warpAffine(align_img, affine_matrix, (cols, rows))
                    results.append(torch.from_numpy(reg_img).permute(2, 0, 1))
                else:
                    results.append(torch.from_numpy(align_img).permute(2, 0, 1))
            else:
                results.append(torch.from_numpy(align_img).permute(2, 0, 1))
        return torch.stack(results, dim=0).to(self.device), torch.from_numpy(ref_img_cpu).permute(2, 0, 1).unsqueeze(0).to(self.device)


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
        choose_strategy = {
            'screw': 'full',
            'metal_nut': 'full',
            **{name: 'unregistered' if name in TEXTURE_CLASS_NAMES else 'translation' for name in CLASS_NAMES if
               name not in ['screw', 'metal_nut']}
        }
        for class_name in dataset_class_names:
            class_name = 'metal_nut'
            print(class_name)
            if choose == 'mvtec':
                cls_dataloader = build_custom_dataloader(cfg_dataset, False, False, class_name=class_name)
            elif choose == 'visa':
                cls_dataloader = build_visa_dataloader(cfg_dataset, False, False, class_name=class_name)
            else:
                cls_dataloader = build_bt_dataloader(cfg_dataset, False, False, class_name=class_name)

            data_len = len(cls_dataloader)
            backup_data = None
            first_img = None
            col_img = []
            col_reg_img = []
            col_time = []

            start_time = time.time()
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
                    model = FastReg(input_images.size(0) - 1, 224, 'reflection', _device,
                                    use_rotation=True if choose_strategy[class_name] == 'full' else False)
                    registered, reg_model, matrix = model.fast_registration(first_img, input_images[
                        indices != most_similar_index])
                    registered, _ = reg_model.affine_trans(matrix, input_images[
                        indices != most_similar_index], padding_mode='zeros')
                    registered_images = torch.cat([first_img, registered.detach()], dim=0)

                    # model = CustomReg(input_images.size(0) - 1, device=_device)
                    # registered = model.register_with_sift(first_img, input_images[
                    #     indices != most_similar_index])
                    # # registered = model(input_images[indices != most_similar_index])
                    # registered_images = torch.cat([registered[1], registered[0]], dim=0)

                else:
                    model = FastReg(input_images.size(0), 224, 'reflection', _device,
                                    use_rotation=True if choose_strategy[class_name] == 'full' else False)
                    registered, reg_model, matrix = model.fast_registration(first_img, input_images)
                    registered, _ = reg_model.affine_trans(matrix, input_images, padding_mode='zeros')
                    registered_images = registered.detach()
                    # model = CustomReg(input_images.size(0), device=_device)
                    # registered = model.register_with_sift(first_img, input_images)
                    # # registered_images = model(input_images)
                    # registered_images = registered[0]
                col_img.append(input_images.cpu())
                col_reg_img.append(registered_images.cpu())
            print(time.time() - start_time)
            all_registered = torch.cat(col_reg_img, dim=0)
            all_images = torch.cat(col_img, dim=0)
            from tool.visualization import show_batch_images
            # show_batch_images(all_images)
            # show_batch_images(all_registered, info='')
            show_batch_images(all_registered)
            break
            # save_dir = os.path.join('reg_res', resume)
            # os.makedirs(save_dir, exist_ok=True)
            # lib.save_metrics_to_csv(save_dir, metrics)


    test_one()
