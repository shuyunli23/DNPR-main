import os
import csv
import re
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score, 
    auc
    )
from skimage import measure


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