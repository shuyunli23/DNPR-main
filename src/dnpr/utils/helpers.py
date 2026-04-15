import torch
import os, cv2
import random
import numpy as np


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
    Select the n features with the lowest scores from the input features.

    Parameters:
        features (Tensor): A tensor of shape (b, c, h, w) representing the features.
        scores (Tensor): A tensor of shape (b, h, w) representing the scores.
        n (int): The number of features to select.

    Returns:
        Tuple[Tensor, Tensor]:
            - selected_features: A tensor of shape (n, c, h, w) containing the features with the lowest scores.
            - selected_scores: A tensor of shape (n, h, w) containing the corresponding scores.
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
    Select a specified number of features from the input features based on batch-wise ranking.

    If the batch size b is less than or equal to n, all features and scores are returned.
    Otherwise, the features are ranked by their scores, and n features are selected from the middle of the ranking.

    Parameters:
        features (Tensor): A tensor of shape (b, c, h, w) representing the features.
        scores (Tensor): A tensor of shape (b, h, w) representing the scores.
        n (int): The number of features to select.

    Returns:
        Tuple[Tensor, Tensor]:
            - selected_features: A tensor of shape (n, c, h, w) containing the selected features.
            - selected_scores: A tensor of shape (n, h, w) containing the corresponding scores.
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