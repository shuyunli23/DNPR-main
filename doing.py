# import torch
# import time
# from lib.common import F
#
# def set_torch_device(gpu_ids):
#     """Returns correct torch.device.
#
#     Args:
#         gpu_ids: [list] list of gpu ids. If empty, cpu is used.
#     """
#     if len(gpu_ids):
#         # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
#         return torch.device("cuda:{}".format(gpu_ids[0]))
#     return torch.device("cpu")
#
#
# def get_gpu_memory_usage():
#     allocated = torch.cuda.memory_allocated()
#     reserved = torch.cuda.memory_reserved()
#     return allocated, reserved
# def apply_neighbor_unfold(feature_map, nbr):
#     """
#     Expands the spatial dimensions of the input feature map by unfolding neighbors.
#
#     Parameters:
#         feature_map (torch.Tensor): Input tensor of shape (b-1, c, h, w).
#         nbr (int): Size of the neighborhood to consider.
#
#     Returns:
#         torch.Tensor: Transformed tensor of shape (b-1, c, h, w, nbr, nbr).
#     """
#     # Calculate the half size of the neighborhood
#     half_nbr = int(nbr / 2)
#
#     # Apply reflection padding
#     padded_feature_map = F.pad(feature_map,
#                                (half_nbr, half_nbr, half_nbr, half_nbr),
#                                mode='reflect')
#
#     # Unfold along the height and width dimensions
#     unfolded_feature_map = padded_feature_map.unfold(2, nbr, 1).unfold(3, nbr, 1)
#
#     return unfolded_feature_map
#
#
# def look_gpu(device):
#     allocated_memory = torch.cuda.memory_allocated(device)
#     reserved_memory = torch.cuda.memory_reserved(device)
#
#     allocated_memory_gb = allocated_memory / (1024 ** 3)
#     reserved_memory_gb = reserved_memory / (1024 ** 3)
#
#     print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")
#     print(f"Reserved Memory: {reserved_memory_gb:.2f} GB")
#     print('\n-----\n')
#
#
# def test_one(device):
#     for _ in range(1000):
#         x = torch.randn((1, 256, 42, 42), device=device)
#         y = torch.randn((4, 256, 42 * 42), device=device)
#         y = y.unsqueeze(2)
#         _, c, h, w = x.shape
#         knowledge_sores = torch.min(torch.norm(x.reshape(1, c, -1).unsqueeze(-1) - y, dim=1), -1)[0].reshape(4, h, w)
#
#
# def test_two(device):
#     xx = torch.randn((100, 256, 42, 42), device=device)
#     for _ in range(2):
#         look_gpu(device)
#         x = torch.randn((1, 256, 42, 42), device=device)
#         look_gpu(device)
#         y = torch.randn((4, 256, 42 * 42), device=device)
#         look_gpu(device)
#         x2 = torch.randn((100, 256, 42, 42), device=device)  # (b-1, c, h, w)
#         nbr = 3
#         look_gpu(device)
#         # (n, c, h, w) -> (n, c, h, w, nbr, nbr)
#         # x2 = apply_neighbor_unfold(x2, nbr)
#         # look_gpu(device)
#         # # (n, c, h, w, nbr, nbr) -> (n, c, h*nbr, w*hbr)
#         #
#         # # x2= (
#         # #         x.unsqueeze(-1).unsqueeze(-1) - x2
#         # # ).permute(0, 1, 2, 4, 3, 5).reshape(100, 256, 42 * nbr, 42 * nbr)
#         # # look_gpu(device)
#         #
#         # # calculate distance
#         # # (n, c, h*nbr, w*hbr) -> (n, h*nbr, w*nbr) -> (n, h, w) -> (h, w)
#         # batch_scores = -F.max_pool2d(-torch.norm((
#         #         x.unsqueeze(-1).unsqueeze(-1) - x2
#         # ).permute(0, 1, 2, 4, 3, 5).reshape(100, 256, 42 * nbr, 42 * nbr), dim=1), kernel_size=nbr, stride=nbr)
#         # y = y.unsqueeze(2)
#         batch_scores = process_batches(x, x2, nbr, 4)
#         _, c, h, w = x.shape
#         batch_size = 1  # 根据显存情况调整
#         knowledge_scores = []
#         look_gpu(device)
#         # for i in range(0, y.shape[0], batch_size):
#         #     y_batch = y[i:i + batch_size]
#         #     scores = torch.norm(x.reshape(1, c, -1).unsqueeze(-1) - y_batch, dim=1)
#         #     knowledge_scores.append(torch.min(scores, -1)[0])
#         #
#         # knowledge_scores = torch.cat(knowledge_scores, dim=0).reshape(-1, h, w)
#         look_gpu(device)
#         print('\n++++++++++++++++++\n')
#
#
# def process_batches(x, x2, nbr, batch_size):
#     b2 = x2.shape[0]  # x2的批量大小
#     batch_scores_list = []
#
#     for start in range(0, b2, batch_size):
#         end = min(start + batch_size, b2)
#         x2_batch = x2[start:end]  # 获取当前批次
#
#         # 应用邻域展开
#         x2_batch_unfolded = apply_neighbor_unfold(x2_batch, nbr)
#
#         # 计算距离
#         error = (x.unsqueeze(-1).unsqueeze(-1) - x2_batch_unfolded).permute(0, 1, 2, 4, 3, 5)
#         error_reshaped = error.reshape(end - start, 256, 42 * nbr, 42 * nbr)
#
#         # 计算 batch_scores
#         batch_scores = -F.max_pool2d(-torch.norm(error_reshaped, dim=1), kernel_size=nbr, stride=nbr)
#         batch_scores_list.append(batch_scores)
#
#     # 合并所有批次的得分
#     return torch.cat(batch_scores_list, dim=0)
#
#
# st_time = time.time()
# device = set_torch_device([2])
# # test_one(device)
# with torch.no_grad():
#     test_two(device)
# end_time = time.time()
# print(f"Speed: {(end_time - st_time) / 60:.2f}")
import torch
from lib.utils import fix_seeds


# def select_features(features, scores, n=2):
#     """
#     Select the features with the lowest scores from the given features.
#
#     Parameters:
#         features (Tensor): A tensor of shape (b, c, h, w) representing features.
#         scores (Tensor): A tensor of shape (b, h, w) representing the scores.
#         n (int): The number of features to select.
#
#     Returns:
#         Tensor: A tensor of shape (n, c, h, w) containing the selected features.
#     """
#     # Ensure the input shapes are correct
#     assert features.dim() == 4, "features should be a 4D tensor with shape (b, c, h, w)"
#     assert scores.dim() == 3, "scores should be a 3D tensor with shape (b, h, w)"
#     b, c, h, w = features.shape
#
#     # Get the indices of the lowest n scores
#     _, indices = torch.topk(scores.view(b, -1), n, dim=0, largest=False)
#
#     # Compute the indices for the features
#     num_features = h * w
#     indices = indices * num_features + torch.arange(num_features, device=features.device).view(-1)
#
#     # Extract features
#     features_reshaped = features.permute(0, 2, 3, 1).reshape(-1, c)  # (b, c, h, w) -> (b*h*w, c)
#     selected_features = features_reshaped[indices].clone()  # (n*h*w, c)
#
#     scores_reshaped = scores.reshape(-1, 1)
#     selected_scores = scores_reshaped[indices].clone()
#
#     return selected_features.reshape(n, h, w, c).permute(0, 3, 1, 2), selected_scores.reshape(n, h, w)
# fix_seeds(0)
# f = torch.randint(0, 9, (4, 2, 5, 5))
# s = torch.randint(0, 9, (4, 5, 5))
# res_f, res_s = select_features(f, s)
# print(f, s)
# print(res_f)
# print(res_s)

# total_recall = [None, None, None, None]
# total_recall[1] = torch.tensor([[1, 2], [3, 4]])  # 赋值一个张量
#
# print(total_recall)
# def get_identity_matrix(batch_size):
#     theta = torch.zeros(batch_size, 1, dtype=torch.float32, requires_grad=False)
#     translation = torch.zeros((batch_size, 2), dtype=torch.float32, requires_grad=False)
#     cos_theta, sin_theta = torch.ones_like(theta), torch.zeros_like(theta)
#     t_x, t_y = translation[:, 0].unsqueeze(1), translation[:, 1].unsqueeze(1)
#     row1 = torch.stack([cos_theta, -1 * sin_theta, t_x], dim=1)
#     row2 = torch.stack([sin_theta, cos_theta, t_y], dim=1)
#     identity_matrix = torch.stack([row1, row2], dim=1).squeeze(3)
#
#     return identity_matrix
# print(get_identity_matrix(1))
import numpy as np
list1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
list2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
list3 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
list4 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 将所有列表放入一个数组
data_arrays = np.array([list1, list2, list3, list4])

# 计算每个位置的均值和方差
mean_values = np.mean(data_arrays, axis=0)
variance_values = np.var(data_arrays, axis=0)
