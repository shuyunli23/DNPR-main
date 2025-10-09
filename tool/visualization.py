import torch
import os
import matplotlib.pyplot as plt
import math
from lib.utils import denormalization, normalize_data, np
from sklearn.manifold import TSNE
import matplotlib.cm as cm


def show_image(image, figsize=(8, 8), cmap=None, save_dir=''):
    """
    Display or save a single image. Supports PyTorch Tensor and NumPy array formats,
    for RGB (3 channels) or single-channel grayscale images.

    Args:
        image (torch.Tensor or numpy.ndarray): Input image data. Should have shape (C, H, W), (H, W, C), or (H, W).
        figsize (tuple, optional): Figure size, default is (8, 8).
        cmap (str, optional): Colormap for grayscale images. Defaults to None, using automatic colormap.
        save_dir (str, optional): Path to save the image. If empty, displays the image directly.

    Raises:
        TypeError: If input is not a PyTorch Tensor or a NumPy array.
        ValueError: If image dimensions do not match expected format.
    """
    # Ensure input is a Tensor or NumPy array
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise TypeError("Input must be a PyTorch Tensor or a NumPy array.")

    # Convert PyTorch Tensor to NumPy array if necessary
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()  # Detach and move to CPU if it's a tensor
        if image.dim() == 3:  # Tensor format should be (C, H, W)
            image = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        elif image.dim() == 2:  # Grayscale format (H, W)
            image = image.numpy()
        else:
            raise ValueError("Tensor must have 2 (H, W) or 3 (C, H, W) dimensions.")

    # Ensure image has correct dimensions after conversion
    if image.ndim == 3 and image.shape[2] not in [1, 3]:
        raise ValueError("Image must have shape (H, W, C) for RGB or grayscale (H, W) format.")
    elif image.ndim == 2:  # Expand grayscale (H, W) to (H, W, 1) for uniform handling
        image = np.expand_dims(image, axis=-1)

    # Normalize the image to [0, 1] range if necessary
    img_min, img_max = image.min(), image.max()
    if img_min < 0 or img_max > 1:  # Normalize only if values are outside [0, 1]
        if img_max <= 255:
            image = image / 255.0
        else:
            image = (image - img_min) / (img_max - img_min)

    # Display or save the image
    plt.figure(figsize=figsize)
    plt.imshow(image.squeeze(), cmap=cmap)  # Use squeeze to handle grayscale images with (H, W, 1)
    plt.axis("off")

    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def show_batch_images(tensor_batch, rows=None, cols=None, need_de_normalize=False, save_dir=''):
    """
    Display or save a batch of images from a tensor.

    Args:
        tensor_batch (torch.Tensor): A tensor of shape (batch_size, channels, height, width).
        rows (int, optional): Number of rows in the displayed grid. Calculated if not specified.
        cols (int, optional): Number of columns in the displayed grid. Calculated if not specified.
        need_de_normalize (bool, optional): Whether to de-normalize the images before displaying. Default is False.
        save_dir (str, optional): Path to save the image grid. If empty, displays the images directly.

    Returns:
        None
    """

    # Move the tensor to CPU and detach it from the computation graph
    tensor_batch = tensor_batch.detach().cpu()

    # If rows and cols are not specified, calculate them based on batch size
    if rows is None and cols is None:
        num_images = tensor_batch.shape[0]
        rows = int(math.sqrt(num_images))
        cols = math.ceil(num_images / rows)

    # Create a canvas to display the images
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3))

    if need_de_normalize:
        for i, ax in enumerate(axes.flat):
            if i < tensor_batch.shape[0]:
                image = denormalization(tensor_batch[i].numpy())
                ax.imshow(image)
            ax.axis('off')
    else:
        tensor_batch = normalize_data(tensor_batch).permute(0, 2, 3, 1)
        for i, ax in enumerate(axes.flat):
            if i < tensor_batch.shape[0]:
                image = tensor_batch[i].numpy()
                ax.imshow(image)
            ax.axis('off')

    plt.tight_layout()
    # Save or display the grid
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

    plt.close(fig)


def plot_histogram(data1, data2=None, bins=20, label1='Data 1', label2='Data 2', is_density=False):
    """
    Plot histograms of two datasets. The second dataset is shown in orange.

    Parameters:
    data1 -- The first tensor (e.g., torch.Size([16, 28, 28]))
    data2 -- The second tensor (optional)
    bins -- Number of bins for the histograms
    label1 -- Label for the first dataset
    label2 -- Label for the second dataset
    """

    # Check if data1 is on GPU and move it to CPU
    if data1.is_cuda:
        data1 = data1.cpu()

    # Flatten and normalize the first dataset
    flat_data1 = data1.view(-1)
    normalized1 = normalize_data(flat_data1)

    # Plot the histogram for the first dataset
    plt.figure(figsize=(10, 6))
    counts1, bins1 = np.histogram(normalized1.numpy(), bins=bins, density=is_density)
    plt.bar(bins1[:-1], counts1, width=np.diff(bins1), alpha=0.5, label=label1, edgecolor='none')

    # If a second dataset is provided
    if data2 is not None:
        # Check if data2 is on GPU and move it to CPU
        if data2.is_cuda:
            data2 = data2.cpu()

        # Flatten and normalize the second dataset
        flat_data2 = data2.view(-1)
        normalized2 = normalize_data(flat_data2)

        # Plot the histogram for the second dataset
        counts2, bins2 = np.histogram(normalized2.numpy(), bins=bins, density=is_density)
        plt.bar(bins2[:-1], counts2, width=np.diff(bins2), alpha=0.5, color='orange', label=label2, edgecolor='none')

        # Calculate overlap
        overlap = np.minimum(counts1, counts2)
        plt.bar(bins1[:-1], overlap, width=np.diff(bins1), alpha=0.5, color='red', label='Overlap', edgecolor='none')

    bar_width = np.diff(bins1)
    plt.title('Histogram of Scores')
    plt.xlabel('Normalized Score')
    plt.ylabel('Frequency')
    plt.xlim(-bar_width[0]/2, 1+bar_width[-1]/2)  # Set the x-axis limits
    plt.legend()
    plt.show()


def plot_loss(loss_list, loss_dir=''):
    plt.figure()
    plt.plot([i for i in range(1, len(loss_list) + 1)], loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    if loss_dir == '':
        plt.show()
    else:
        os.makedirs(loss_dir, exist_ok=True)
        plt.savefig(f'{loss_dir}/curve_plot.png')


def plot_anomaly_scores_histogram(all_scores_pixel, gt_labels, y_limit=(0, 1e6), alpha=0.5):
    """
    绘制异常分数的直方图。

    参数:
    - all_scores_pixel: 形状为 (83, 288, 288) 的 PyTorch Tensor，表示异常分数。
    - gt_labels: 形状为 (83, 1, 288, 288) 的 PyTorch Tensor，表示标签（0或1）。
    - y_limit: 纵坐标范围的元组，默认为 (0, 1e6)。
    - alpha: 直方图的透明度，默认为 0.5。
    """
    # 将 Tensor 展平为一维数组
    flattened_scores = all_scores_pixel.view(-1)
    flattened_labels = gt_labels.view(-1)

    # 提取异常和正常的分数
    normal_scores = flattened_scores[flattened_labels == 0]
    anomalous_scores = flattened_scores[flattened_labels == 1]

    # 计算重叠区域
    bins = np.histogram_bin_edges(flattened_scores.numpy(), bins=50)
    normal_hist, _ = np.histogram(normal_scores.numpy(), bins=bins)
    anomalous_hist, _ = np.histogram(anomalous_scores.numpy(), bins=bins)

    overlap_hist = np.minimum(normal_hist, anomalous_hist)

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores.numpy(), bins=50, color='blue', alpha=alpha, label='Normal (Label: 0)')
    plt.hist(anomalous_scores.numpy(), bins=50, color='orange', alpha=alpha, label='Anomalous (Label: 1)')
    plt.bar(bins[:-1], overlap_hist, width=np.diff(bins), color='red', alpha=0.5, label='Overlap')

    plt.title('Histogram of Anomaly Scores')
    plt.xlabel('Anomaly Scores')
    plt.ylabel('Frequency')

    # 设置 y 轴的范围
    plt.ylim(y_limit)

    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def visualize_tensor_tsne(tensor, random_state=42, marker_size=5, title='t-SNE Visualization'):
    """
    Visualize a 4D tensor with shape (b, c, h, w) by reshaping it to (b*h*w, c),
    applying t-SNE for dimensionality reduction to 2D, and plotting the result.

    Parameters:
        tensor (np.ndarray): Input tensor with shape (b, c, h, w).
        random_state (int): Random seed for t-SNE.
        marker_size (int): Size of scatter plot markers.
        title (str): Title of the plot.
    """
    if tensor.ndim != 4:
        raise ValueError("Input tensor must be 4-dimensional (b, c, h, w)")

    b, c, h, w = tensor.shape
    # Reshape to (b*h*w, c)
    reshaped = tensor.transpose(0, 2, 3, 1).reshape(-1, c)

    # Apply t-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=random_state)
    data_2d = tsne.fit_transform(reshaped)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], s=marker_size)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def visualize_tensor_tsne_colored(tensor, random_state=42, marker_size=5, title='t-SNE Visualization'):
    """
    Visualize a 4D tensor with shape (b, c, h, w) by reshaping it to (b*h*w, c),
    applying t-SNE for dimensionality reduction to 2D, and plotting the result.
    Each batch (b) is displayed in a different color.

    Parameters:
        tensor (np.ndarray): Input tensor with shape (b, c, h, w).
        random_state (int): Random seed for t-SNE.
        marker_size (int): Size of scatter plot markers.
        title (str): Title of the plot.
    """
    if tensor.ndim != 4:
        raise ValueError("Input tensor must be 4-dimensional (b, c, h, w)")

    b, c, h, w = tensor.shape
    # Prepare a colormap with b different colors
    colors = cm.get_cmap("tab10", b)

    plt.figure(figsize=(8, 6))
    tsne = TSNE(n_components=2, random_state=random_state)

    # For each batch, extract features, reshape and apply t-SNE
    for i in range(b):
        # Shape (c, h, w) -> (h, w, c) -> (h*w, c)
        features = tensor[i].transpose(1, 2, 0).reshape(-1, c)
        # t-SNE for current batch; 如果数据量不大，也可以将所有 batch 一起降维再分色
        data_2d = tsne.fit_transform(features)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=marker_size, color=colors(i), label=f'Batch {i}')

    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()




