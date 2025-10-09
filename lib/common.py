import torch
import copy
import math
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import nn


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                # The backbone will throw an Exception once it reached the last
                # layer to compute features from. Computation will stop there.
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

    def feature_resolutions(self, input_shape):
        """Computes the feature resolutions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [[_output[layer].shape[2], _output[layer].shape[3]] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        # if self.raise_exception_to_break:
        #     raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)

    @staticmethod
    def _reduce(features):
        pass


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim, keep_shape=False):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim
        self.keep_shape = keep_shape

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        output = F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)
        if self.keep_shape:
            output = output.unsqueeze(2).unsqueeze(3)
        return output


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class FeatureMapProjection(torch.nn.Module):
    """
    Random linear projection.
    """
    def __init__(self, in_channels, out_channels):
        super(FeatureMapProjection, self).__init__()

        self.projection = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        # initialize projection parameters
        torch.nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.projection.bias, 0)

        # fixed projection parameters
        for param in self.projection.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.projection(x)
        return output


class RescaleSegment:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)

            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )

            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class PatchMaker:
    def __init__(self, patch_size, stride=None):
        self.patch_size = patch_size
        self.stride = stride

    def make_patch(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            :param features:  [torch.Tensor, bs x c x w x h]
            :param return_spatial_info: Default to False
        Returns:
            unfolded_features: [torch.Tensor, bs * w // stride * h // stride, c, patch size,
            patch size]
        """
        padding = int((self.patch_size - 1) / 2)
        unfold_er = torch.nn.Unfold(
            kernel_size=self.patch_size, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfold_er(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patch_size - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patch_size, self.patch_size, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @staticmethod
    def patch_alignment(features, patch_shapes):
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        return features

    @staticmethod
    def recombinant_features(input_features, batch_size):
        dim = input_features.size(1)
        batch_features = input_features.reshape(batch_size, -1, dim)
        patch_num_sqrt = int(np.sqrt(batch_features.size(1)))
        batch_patch_features = batch_features.reshape(batch_size,
                                                      patch_num_sqrt, patch_num_sqrt,
                                                      dim)
        return batch_patch_features.permute(0, 3, 1, 2)

    @staticmethod
    def apply_neighbor_unfold(feature_map, nbr):
        """
        Expands the spatial dimensions of the input feature map by unfolding neighbors.

        Parameters:
            feature_map (torch.Tensor): Input tensor of shape (b-1, c, h, w).
            nbr (int): Size of the neighborhood to consider.

        Returns:
            torch.Tensor: Transformed tensor of shape (b-1, c, h, w, nbr, nbr).
        """
        # Calculate the half size of the neighborhood
        half_nbr = int(nbr / 2)

        # Apply reflection padding
        padded_feature_map = F.pad(feature_map,
                                   (half_nbr, half_nbr, half_nbr, half_nbr),
                                   mode='reflect')

        # Unfold along the height and width dimensions
        unfolded_feature_map = padded_feature_map.unfold(2, nbr, 1).unfold(3, nbr, 1)

        return unfolded_feature_map


class FeatureProjection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(FeatureProjection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Conv2d(_in, _out, kernel_size=(1, 1), stride=(1, 1), padding=0))
            if i < n_layers - 1:
                if layer_type > 0:
                    self.layers.add_module(f"{i}bn",
                                           torch.nn.InstanceNorm2d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))

    def forward(self, x):

        x = self.layers(x)
        return x


class AffineTrans(nn.Module):
    def __init__(self, batch_size, img_shape=224, mask_method='circular', padding_mode='border', edge_ratio=0.2,
                 is_affine=False):
        super(AffineTrans, self).__init__()
        self.edge_ratio = edge_ratio
        self.padding_mode = padding_mode
        self.batch_size = batch_size
        self.is_affine = is_affine
        self.use_rotation = True

        if is_affine:
            self.matrix = nn.Parameter(torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1), requires_grad=True)
            self.theta, self.translation = None, None
        else:
            # Define the theta and translation
            self.theta = nn.Parameter(torch.zeros(batch_size, 1, dtype=torch.float32))
            self.translation = nn.Parameter(torch.zeros((batch_size, 2), dtype=torch.float32))
            self.matrix = None
        # Make mask
        self.img_shape = img_shape
        if mask_method == 'circular':
            self.mask = self.generate_circular_mask()
        else:
            self.mask = self.generate_four_corner_mask()

    def fixed_parameters(self, fixed_rotation=False, fixed_translation=False):
        if fixed_rotation:
            self.theta.requires_grad = False
            self.use_rotation = False
        if fixed_translation:
            self.translation.requires_grad = False

    def generate_circular_mask(self):
        mask = torch.ones(1, 1, self.img_shape, self.img_shape, dtype=torch.float32)
        center_x = int(self.img_shape / 2)
        center_y = center_x
        side_length = int(self.img_shape * self.edge_ratio)
        radius = (center_y - side_length / 2) * math.sqrt(2)
        y, x = torch.meshgrid(torch.arange(self.img_shape), torch.arange(self.img_shape), indexing='ij')
        mask[:, :, (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2] = 0
        return mask

    def generate_four_corner_mask(self):
        mask = torch.ones(1, 1, self.img_shape, self.img_shape, dtype=torch.float32)
        side_length = int(self.img_shape * self.edge_ratio)

        # Mask 4 corners
        for i in range(side_length):
            mask[:, :, :side_length - i, i] = 0.0  # upper left
            mask[:, :, i, -side_length + i:] = 0.0  # upper right
            mask[:, :, -side_length + i:, i] = 0.0  # lower left
            mask[:, :, -i, -side_length + i:] = 0.0  # lower right

        return mask

    def get_affine_matrix(self):
        if self.is_affine:
            matrix = self.matrix
        else:
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
    def get_identity_matrix(batch_size):
        return torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1)

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
        if self.is_affine:
            matrix = self.matrix
        else:
            matrix = self.get_affine_matrix()
        x_transformed, blank_mask = self.affine_trans(matrix, x, padding_mode=self.padding_mode)
        return x_transformed, blank_mask


if __name__ == '__main__':
    from lib import utils
    from torchsummary import summary
    from lib.backbones import CustomVisualEncoder
    import warnings
    import open_clip

    warnings.filterwarnings("ignore")


    def test_one(model_name="RN50"):
        gpu = [0]
        _device = utils.set_torch_device(gpu)

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained='openai', device=_device
        )
        print(model.visual)
        bone = CustomVisualEncoder(model.visual)
        print(bone.conv1.weight)

        data = torch.randn(8, 3, 288, 288).to(_device)
        _ = bone(data)
        summary(bone, (3, 224, 224), 8)
        feature_aggregator_net = NetworkFeatureAggregator(
            bone, ['layer1', 'layer2', 'layer3', 'layer4'], _device
        )
        _features = feature_aggregator_net(data)
        print(_features['layer1'].shape, _features['layer2'].shape,
              _features['layer3'].shape, _features['layer4'].shape)


    test_one("RN50x4")



