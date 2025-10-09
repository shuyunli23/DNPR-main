import torch
import os, cv2
import random
import csv
import re
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc
from skimage import morphology
from skimage.segmentation import mark_boundaries
from skimage import measure


def select_reference_image_with_orb(images, scale_factor=0.5):
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptors_list = []

    # Convert tensor images to numpy and extract features
    for img in images:
        # Convert tensor to numpy array and reshape
        img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Change from (C, H, W) to (H, W, C)
        resized_img = cv2.resize(img_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

        # Detect ORB keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray_img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Initialize matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Calculate matches between descriptors
    match_counts = []
    for i in range(len(descriptors_list)):
        matches = []
        for j in range(len(descriptors_list)):
            if i != j and descriptors_list[i] is not None and descriptors_list[j] is not None:
                match = bf.match(descriptors_list[i], descriptors_list[j])
                matches.append(len(match))
            else:
                matches.append(0)
        match_counts.append(matches)

    # Sum up matches for each image and find the one with the most matches
    match_sums = np.sum(match_counts, axis=1)
    reference_index = np.argmax(match_sums)

    return reference_index


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


def select_features(features, scores, n=2):
    """
    Select the features with the lowest scores from the given features.

    Parameters:
        features (Tensor): A tensor of shape (b, c, h, w) representing features.
        scores (Tensor): A tensor of shape (b, h, w) representing the scores.
        n (int): The number of features to select.

    Returns:
        Tensor: A tensor of shape (n, c, h, w) containing the selected features.
    """
    # Ensure the input shapes are correct
    assert features.dim() == 4, "features should be a 4D tensor with shape (b, c, h, w)"
    assert scores.dim() == 3, "scores should be a 3D tensor with shape (b, h, w)"
    b, c, h, w = features.shape

    # Get the indices of the lowest n scores
    selected_scores, indices = torch.topk(scores.view(b, -1), n, dim=0, largest=False)

    # Compute the indices for the features
    num_features = h * w
    indices = indices * num_features + torch.arange(num_features, device=features.device).view(-1)

    # Extract features
    features_reshaped = features.permute(0, 2, 3, 1).reshape(-1, c)  # (b, c, h, w) -> (b*h*w, c)
    selected_features = features_reshaped[indices].clone()  # (n*h*w, c)

    return selected_features.reshape(n, h, w, c).permute(0, 3, 1, 2), selected_scores.reshape(n, h, w)


def feature_select(features, scores, n=2):
    """
    Select the features with the lowest scores from the given features.

    Parameters:
        features (Tensor): A tensor of shape (b, c, h, w) representing features.
        scores (Tensor): A tensor of shape (b, h, w) representing the scores.
        n (int): The number of features to select.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - Tensor of shape (n, c, h, w): The selected features.
            - Tensor of shape (n, h, w): The selected scores.
    """
    # Ensure the input shapes are correct
    assert features.dim() == 4, "features should be a 4D tensor with shape (b, c, h, w)"
    assert scores.dim() == 3, "scores should be a 3D tensor with shape (b, h, w)"
    b, c, h, w = features.shape
    if b <= n:
        selected_scores = scores
        selected_features = features
    else:
        start_idx = (b - n) // 2
        end_idx = start_idx + n

        selected_scores, indices = torch.topk(scores.view(b, -1), end_idx + 1, dim=0, largest=False)

        selected_scores = selected_scores[start_idx:end_idx]
        selected_scores = selected_scores.reshape(n, h, w)
        indices = indices[start_idx:end_idx]

        # Compute the indices for the features
        num_features = h * w
        indices = indices * num_features + torch.arange(num_features, device=features.device).view(-1)

        # Extract features
        features_reshaped = features.permute(0, 2, 3, 1).reshape(-1, c)  # (b, c, h, w) -> (b*h*w, c)
        selected_features = features_reshaped[indices]  # (n*h*w, c)
        selected_features = selected_features.reshape(n, h, w, c).permute(0, 3, 1, 2)

    return selected_features, selected_scores


def batch_center_crop(input_tensor, target_size):
    """
    Center crop a batch of tensors to the specified target size.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor to be cropped (shape: N, C, H, W).
    - target_size (tuple): The target size (height, width) for the crop.

    Returns:
    - torch.Tensor: The center-cropped tensor.
    """
    # Get the dimensions of the input tensor
    N, C, h, w = input_tensor.shape

    # Ensure the target size is reasonable
    target_height, target_width = target_size
    if target_height > h or target_width > w:
        raise ValueError("Target size must be smaller than the input dimensions.")

    # Calculate the starting and ending positions for cropping
    start_h = (h - target_height) // 2
    end_h = start_h + target_height
    start_w = (w - target_width) // 2
    end_w = start_w + target_width

    # Crop the tensor using slicing
    cropped_tensor = input_tensor[:, :, start_h:end_h, start_w:end_w].clone()

    return cropped_tensor


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def normalize_data(data):
    if isinstance(data, np.ndarray):
        max_values = np.max(data)
        min_values = np.min(data)
        normalized_data = (data - min_values) / (max_values - min_values)
    elif isinstance(data, torch.Tensor):
        max_values = torch.max(data)
        min_values = torch.min(data)
        normalized_data = (data - min_values) / (max_values - min_values)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

    return normalized_data


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def compute_metrics(labels, img_scores, gt_mask, scores, metrics_to_compute=None):
    """
    Computes specified evaluation metrics including AUC, F1-score, IOU, and PRO.

    Args:
        labels: True labels for image-level evaluation.
        img_scores: Scores predicted for each image.
        gt_mask: Ground truth segmentation masks for pixel-level evaluation.
        scores: Predicted scores for pixel-level evaluation.
        metrics_to_compute: List of metrics to compute (
        e.g., ['img_auc', 'pixel_auc', 'img_ap', 'pixel_ap', 'iou', 'pro', 'pixel_f1', 'img_f1', 'fpr_tpr']
        ).

    Returns:
        Dictionary of computed metrics based on metrics_to_compute.
    """
    # Convert inputs to numpy arrays if they are not already
    if metrics_to_compute is None:
        metrics_to_compute = [
            'img_auc', 'pixel_auc', 'img_ap', 'pixel_ap', 'iou', 'pro', 'pixel_f1', 'img_f1', 'fpr_tpr'
        ]
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(img_scores, np.ndarray):
        img_scores = np.array(img_scores)
    if not isinstance(gt_mask, np.ndarray):
        gt_mask = np.array(gt_mask)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    results = {}

    # Image-level ROC AUC score
    if 'img_auc' in metrics_to_compute:
        results['img_auc'] = roc_auc_score(labels, img_scores)

    if 'fpr_tpr' in metrics_to_compute:
        img_fpr, img_tpr, _ = roc_curve(labels, img_scores)
        results['img_fpr'] = img_fpr
        results['img_tpr'] = img_tpr
        pixel_fpr, pixel_tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        results['pixel_fpr'] = pixel_fpr
        results['pixel_tpr'] = pixel_tpr

    # Get optimal threshold for pixel-level evaluation
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    optimal_threshold = thresholds[np.argmax(f1)]
    results['threshold'] = optimal_threshold

    # Pixel-level ROC AUC score
    if 'pixel_auc' in metrics_to_compute:
        results['pixel_auc'] = roc_auc_score(gt_mask.flatten(), scores.flatten())

    # Average Precision (AP) for image-level and pixel-level
    if 'img_ap' in metrics_to_compute:
        results['img_ap'] = average_precision_score(labels, img_scores)
    if 'pixel_ap' in metrics_to_compute:
        results['pixel_ap'] = average_precision_score(gt_mask.flatten(), scores.flatten())

    # Calculate pixel IOU
    if 'iou' in metrics_to_compute:
        intersection = np.logical_and(np.squeeze(gt_mask), (scores > optimal_threshold))
        union = np.logical_or(np.squeeze(gt_mask), (scores > optimal_threshold))
        results['iou'] = np.sum(intersection) / np.sum(union)

    # Calculate Per-Region Overlap (PRO)
    if 'pro' in metrics_to_compute:
        results['pro'] = cal_pro_score(gt_mask.squeeze(), scores)

    # F1-score for pixel-level predictions
    if 'pixel_f1' in metrics_to_compute:
        results['pixel_f1_max'] = np.max(f1[np.isfinite(f1)])

    # Image-wise F1-score calculation
    if 'img_f1' in metrics_to_compute:
        # Calculate precision and recall for the image scores
        precision, recall, thresholds = precision_recall_curve(labels, img_scores)
        a = 2 * precision * recall
        b = precision + recall
        img_f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)  # Avoid division by zero
        results['img_f1_max'] = np.max(img_f1[np.isfinite(img_f1)])  # Get the maximum F1-score

    return results


def plot_segmentation_images(test_img, scores, gts, threshold, save_dir, class_name, plot_more=''):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask <= threshold] = 0
        high_scores = mask * 255
        mask[mask > threshold] = 1
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        alpha = np.where(high_scores == 0, 0.0, 0.3)
        ax_img[4].imshow(vis_img)
        ax_img[4].imshow(high_scores, cmap='hot', alpha=alpha, interpolation='none')
        ax_img[4].title.set_text('Segmentation result')

        plt.axis('off')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

        if plot_more != '':
            fig_img_another, ax_img_another = plt.subplots(1, 4, figsize=(9, 3))
            fig_img.subplots_adjust(right=0.9)
            for ax_j in ax_img_another:
                ax_j.axes.xaxis.set_visible(False)
                ax_j.axes.yaxis.set_visible(False)
            ax_img_another[0].imshow(img)
            ax_img_another[0].title.set_text('Image')
            ax_img_another[1].imshow(gt, cmap='gray')
            ax_img_another[1].title.set_text('GroundTruth')
            ax_img_another[2].imshow(mask, cmap='gray')
            ax_img_another[2].title.set_text('Predicted mask')

            # Create an array of the same size as the original image to store the superimposed image
            overlay = np.zeros_like(img)
            # Set false positives to red
            overlay[np.logical_and(gt != 1, mask == 255)] = [255, 0, 0]
            # Set true positive as green
            overlay[np.logical_and(gt == 1, mask == 255)] = [0, 255, 0]
            # Set false negatives to blue
            overlay[np.logical_and(gt == 1, mask != 255)] = [0, 0, 255]
            ax_img_another[3].imshow(cv2.addWeighted(img, 1, overlay, 0.5, 0))
            ax_img_another[3].title.set_text('Overlay')

            plt.axis('off')
            fig_img_another.savefig(os.path.join(plot_more, class_name + '_overlay_{}'.format(i)), dpi=100)
            plt.close()


def plot_segmentation_images_for_paper(test_img, registered, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        reg = registered[i]
        reg = denormalization(reg)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask <= threshold] = 0
        high_scores = mask * 255
        mask[mask > threshold] = 1
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255

        plt.figure()

        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        # image
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'image_{}'.format(i)), bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        # mask
        plt.imshow(gt, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'mask_{}'.format(i)), bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        # predicted
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'pred_{}'.format(i)), bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        # segmentation
        alpha = np.where(high_scores == 0, 0.0, 0.3)
        plt.imshow(vis_img)
        plt.imshow(high_scores, cmap='hot', alpha=alpha, interpolation='none')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'seg_{}'.format(i)), bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        # registered
        plt.imshow(reg)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'reg_{}'.format(i)), bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        # TP、FP、TN、FN
        # create an array of the same size as the original image to store the superimposed image
        overlay = np.zeros_like(img)
        # Set false positives to red
        overlay[np.logical_and(gt != 1, mask == 255)] = [255, 0, 0]
        # Set true positive as green
        overlay[np.logical_and(gt == 1, mask == 255)] = [0, 255, 0]
        # Set false negatives to blue
        overlay[np.logical_and(gt == 1, mask != 255)] = [0, 0, 255]
        plt.imshow(cv2.addWeighted(img, 1, overlay, 0.5, 0))
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'overlay_{}'.format(i)), bbox_inches='tight', pad_inches=0, dpi=400)
        plt.clf()

        plt.close()


def plot_auroc_curve(save_path, metrics):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_auroc = ax[0]
    fig_pixel_auroc = ax[1]
    total_img_auroc = []
    total_pixel_auroc = []

    for class_name, res in metrics.items():
        fig_img_auroc.plot(res['img_fpr'], res['img_tpr'],
                           label='%s img_AUROC: %.3f' % (class_name, res['img_auc']))

        fig_pixel_auroc.plot(res['pixel_fpr'], res['pixel_tpr'],
                             label='%s pixel_AUROC: %.3f' % (class_name, res['pixel_auc']))
        total_img_auroc.append(res['img_auc'])
        total_pixel_auroc.append(res['pixel_auc'])

    fig_img_auroc.title.set_text('Average image AUROC: %.3f' % np.mean(total_img_auroc))
    fig_img_auroc.legend(loc="lower right")

    fig_pixel_auroc.title.set_text('Average pixel AUROC: %.3f' % np.mean(total_pixel_auroc))
    fig_pixel_auroc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'auroc_curve.png'), dpi=100)

    print("AUC visualization saved to: ", os.path.join(save_path, 'auroc_curve.png'))


def save_metrics(save_path, metrics):
    csv_path = os.path.join(save_path, 'metrics.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write column names
        writer.writerow(['Class name', 'Image AUROC', 'Pixel AUROC',
                         'Image AP', 'Pixel AP'])

        total_img_auroc = []
        total_pixel_auroc = []
        total_img_ap = []
        total_pixel_ap = []
        total_iou = []

        for class_name, res in metrics.items():
            writer.writerow([class_name, round(res['img_auc'] * 100, 1),
                             round(res['pixel_auc'] * 100, 1),
                             round(res['img_ap'] * 100, 1),
                             round(res['pixel_ap'] * 100, 1),
                             round(res['iou'] * 100, 1)])
            total_img_auroc.append(res['img_auc'])
            total_pixel_auroc.append(res['pixel_auc'])
            total_img_ap.append(res['img_ap'])
            total_pixel_ap.append(res['pixel_ap'])
            total_iou.append(res['iou'])

        # Write average row
        writer.writerow(['Average', round(np.mean(total_img_auroc) * 100, 1),
                        round(np.mean(total_pixel_auroc) * 100, 1),
                        round(np.mean(total_img_ap) * 100, 1),
                        round(np.mean(total_pixel_ap) * 100, 1),
                        round(np.mean(total_iou) * 100, 1)])

    print("All metrics saved to: ", csv_path)


def save_metrics_to_csv(save_path, metrics):
    csv_path = os.path.join(save_path, 'metrics.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write column names
        writer.writerow(['Cls-name', 'I-AUROC', 'P-AUROC', 'I-AP', 'P-AP', 'I-F1', 'P-F1', 'PRO', 'Speed'])

        total_img_auroc = []
        total_pixel_auroc = []
        total_img_ap = []
        total_pixel_ap = []
        total_img_f1 = []
        total_pixel_f1 = []
        total_pro = []
        total_time = []

        for class_name, res in metrics.items():
            writer.writerow([class_name,
                             round(res['img_auc'] * 100, 2),
                             round(res['pixel_auc'] * 100, 2),
                             round(res['img_ap'] * 100, 2),
                             round(res['pixel_ap'] * 100, 2),
                             round(res['img_f1'] * 100, 2),
                             round(res['pixel_f1'] * 100, 2),
                             round(res['pro'] * 100, 2),
                             round(res['test_time'], 2)])  # Adjust the precision as needed

            total_img_auroc.append(res['img_auc'])
            total_pixel_auroc.append(res['pixel_auc'])
            total_img_ap.append(res['img_ap'])
            total_pixel_ap.append(res['pixel_ap'])
            total_img_f1.append(res['img_f1'])
            total_pixel_f1.append(res['pixel_f1'])
            total_pro.append(res['pro'])
            total_time.append(res['test_time'])

        # Write average row
        writer.writerow(['Average',
                         round(np.mean(total_img_auroc) * 100, 2),
                         round(np.mean(total_pixel_auroc) * 100, 2),
                         round(np.mean(total_img_ap) * 100, 2),
                         round(np.mean(total_pixel_ap) * 100, 2),
                         round(np.mean(total_img_f1) * 100, 2),
                         round(np.mean(total_pixel_f1) * 100, 2),
                         round(np.mean(total_pro) * 100, 2),
                         round(np.mean(total_time), 2)])  # Adjust the precision as needed

    print("All metrics saved to: ", csv_path)


def aggregate_metrics(base_path, num_times):
    """
    Aggregates metrics from multiple CSV files, calculates the mean and standard deviation
    for each column in each class, and saves the result in the directory one level above the 'seed_{}' folder.

    Parameters:
    base_path (str): The file path template containing 'seed_X' (e.g., './results/exp_one/batch_16/seed_0/0_shot_custom[wideresnet50]/metrics.csv').
    num_times (int): The number of files (seed values from 0 to num_times-1) to aggregate.
    """
    # Modify base_path to include a placeholder for the seed value using regex
    base_path = re.sub(r'seed_\d+', 'seed_{}', base_path)   # \d+ represents one or more numbers (0-9)
    seed_range = range(num_times)  # Seed values from 0 to num_times-1

    # Columns for each class's results
    columns = ['Cls-name', 'I-AUROC', 'P-AUROC', 'I-AP', 'P-AP', 'I-F1', 'P-F1', 'PRO', 'Speed']
    all_data = []  # List to store data from each CSV file

    # Loop through each seed and read the corresponding CSV file
    for seed in seed_range:
        file_path = base_path.format(seed)  # Format file path with the current seed
        try:
            df = pd.read_csv(file_path)  # Read CSV file
            all_data.append(df)  # Append data to the list
        except FileNotFoundError:
            print(f"Warning: File not found for seed {seed}, skipping...")  # Handle missing files

    # If no data was loaded, print an error and exit
    if not all_data:
        print("Error: No data found. Please check file paths.")
        return

    # Calculate mean and standard deviation for each column in each class
    results = []
    for index, row in all_data[0].iterrows():  # Assuming all files have the same structure
        class_name = row['Cls-name']  # Get class name
        stats = [class_name]  # Start list with class name

        # Calculate mean and standard deviation for each column
        for col in columns[1:]:  # Skip the first column (class name)
            values = [df.loc[index, col] for df in all_data if col in df.columns]  # Collect values from each file
            mean = np.mean(values)  # Calculate mean
            std = np.std(values)    # Calculate standard deviation
            stats.append(f"{mean:.1f} ± {std:.1f}")  # Format as "mean ± std"

        results.append(stats)  # Append stats for this class

    # Determine the output directory as one level above the 'seed_{}' folder
    output_dir = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
    match = re.search(r'(\w+_shot_[\w\[\]]+)', base_path)
    output_path = os.path.join(output_dir, f'{match.group(0)}_results.csv')
    replicates_df = pd.DataFrame(results, columns=columns)
    logging.info(f'''Performance: \n{replicates_df.to_markdown(index=False)}
    ''')
    print(f'''Performance: \n{replicates_df.to_markdown(index=False)}
    ''')
    replicates_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"Results file saved as '{output_path}'")


def print_top_k(tensor, k):
    values, indices = torch.topk(tensor.flatten(), k)
    for i in range(k):
        print(f"Value: {values[i]}, Index: {indices[i]}")


def plot_reconstructed_images(original_img, reconstructed_img, save_dir='./reconstructed_img', class_name='_'):
    num = len(original_img)
    for i in range(num):
        img = original_img[i]
        img = denormalization(img)
        rec_img = denormalization(reconstructed_img)

        fig_img, ax_img = plt.subplots(1, 2, figsize=(5, 3))
        fig_img.subplots_adjust(right=0.9)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(rec_img)
        ax_img[1].title.set_text('Reconstructed Image')

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def plot_loss(loss_list, loss_dir):
    plt.figure()
    plt.plot([i for i in range(1, len(loss_list) + 1)], loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    os.makedirs(loss_dir, exist_ok=True)
    plt.savefig(f'{loss_dir}/curve_plot.png')


if __name__ == '__main__':
    # Example usage
    aggregate_metrics('../results/output/batch_16/seed_4/0_shot_ci[wideresnet101]/metrics.csv', 5)