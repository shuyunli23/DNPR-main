"""
DNPR: Zero-Shot Industrial Anomaly Detection via Dynamic Normal Prototype Evolution.

This module contains the main DNPR model implementation.
"""

import math
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import optim

from dnpr.models.common import (
    Aggregator,
    AffineTrans,
    FeatureMapProjection,
    NetworkFeatureAggregator,
    PatchMaker,
    Preprocessing,
    RescaleSegment,
)
from dnpr.utils.helpers import (
    batch_center_crop,
    feature_select,
    normalize_data,
    select_features,
    select_reference_image_with_orb,
)
from dnpr.utils.metrics import compute_metrics
from dnpr.utils.visualization import plot_segmentation_images


class DNPR:
    """
    DNPR: Zero-shot industrial anomaly detection via dynamic normal prototype evolution.
    
    This class implements the DNPR algorithm for industrial anomaly detection,
    supporting both zero-shot and few-shot settings.
    
    Args:
        args: Configuration namespace containing model parameters.
        backbone: Pretrained backbone network for feature extraction.
        layers_to_extract_from: List of layer names to extract features from.
        dataloader: Dictionary containing dataset and class name information.
        device: Torch device for computation.
        input_shape: Input image shape (C, H, W).
        pretrain_embed_dimension: Dimension of pretrained embeddings.
        target_embed_dimension: Target embedding dimension after aggregation.
        patch_size: Size of patches for feature extraction.
        patch_stride: Stride for patch extraction.
        top_k: Number of top scores for image-level scoring.
        proj_dim: Projection dimension for feature reduction.
        strategy: Registration strategy ('remain', 'transform', 'translate').
        first_batch_lower_limit: Minimum samples required for first batch.
    """

    def __init__(
        self,
        args,
        backbone: torch.nn.Module,
        layers_to_extract_from: List[str],
        dataloader: Dict[str, Any],
        device: torch.device,
        input_shape: Tuple[int, int, int],
        pretrain_embed_dimension: int = 1024,
        target_embed_dimension: int = 1024,
        patch_size: int = 3,
        patch_stride: int = 1,
        top_k: int = 9,
        proj_dim: int = 256,
        strategy: str = "transform",
        first_batch_lower_limit: int = 4,
    ):
        self.class_name = dataloader["cls_name"]
        print(f"Class name: {self.class_name}[{strategy}]")
        
        self.args = args
        self.strategy = strategy
        self.first_batch_lower_limit = first_batch_lower_limit
        self.knwldg_dir = os.path.join(
            args.save_path, args.resume, f"{self.class_name}/prior_knowledge"
        )
        self.backbone = backbone.to(device)

        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.k_shot = args.k_shot
        self.dataloader = dataloader
        self.top_k = top_k
        self.nbr = args.nbr

        self.patch_maker = PatchMaker(patch_size, stride=patch_stride)
        self.use_rotation = strategy == "transform"

        # Initialize knowledge for few-shot learning
        if self.k_shot != 0:
            self.knowledge = self.load_knowledge()
            if self.knowledge is None:
                print("Please first use the few_shot_memory method to create knowledge.")
            else:
                n, c, _, _ = self.knowledge["features"].shape
                self.knowledge["features"] = self.knowledge["features"].reshape(
                    n, c, -1
                ).unsqueeze(2)
        else:
            self.knowledge = None

        # Initialize masks
        self.reg_msk = None
        self.edge_msk = self._get_edge_mask(input_shape)

        # Initialize memory banks
        self.glo_memory = None
        self.glo_scores = None
        self.gol_patch_scores = None
        self.glo_memory_num = args.glo_memory_num
        self.loc_memory_num = args.loc_memory_num
        self.loc_memory = None
        self.loc_scores = None

        # Build forward modules
        self.forward_modules = torch.nn.ModuleDict({})
        self._build_modules(input_shape, pretrain_embed_dimension, target_embed_dimension, proj_dim)

    def _build_modules(
        self,
        input_shape: Tuple[int, int, int],
        pretrain_embed_dimension: int,
        target_embed_dimension: int,
        proj_dim: int,
    ) -> None:
        """Build and initialize forward modules."""
        # Feature aggregator
        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # Preprocessing
        preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension

        # Pre-adaptation aggregator
        pre_adapt_aggregator = Aggregator(target_dim=target_embed_dimension)
        self.forward_modules["pre_adapt_aggregator"] = pre_adapt_aggregator.to(self.device)

        # Anomaly segmentation rescaler
        self.anomaly_segment = RescaleSegment(
            device=self.device, target_size=input_shape[-2:]
        )

        # Feature projector
        feat_projector = FeatureMapProjection(target_embed_dimension, proj_dim)
        self.forward_modules["feat_projector"] = feat_projector.to(self.device)

    def save_knowledge(self, images: torch.Tensor, features: torch.Tensor) -> None:
        """Save knowledge (images and features) for few-shot learning."""
        os.makedirs(self.knwldg_dir, exist_ok=True)
        torch.save(
            {"images": images, "features": features},
            os.path.join(self.knwldg_dir, "knowledge.pth"),
        )

    def load_knowledge(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load saved knowledge from the knowledge directory.

        Returns:
            Dictionary containing saved knowledge with keys 'images' and 'features',
            or None if not found.
        """
        knowledge_path = os.path.join(self.knwldg_dir, "knowledge.pth")
        if os.path.exists(knowledge_path):
            return torch.load(knowledge_path, map_location=self.device)
        return None

    def features_extract(
        self, images: torch.Tensor, provide_patch_shapes: bool = False
    ) -> torch.Tensor:
        """
        Extract patch features from images.
        
        Args:
            images: Input images tensor of shape (B, C, H, W).
            provide_patch_shapes: Whether to return patch shapes.
            
        Returns:
            Extracted features, optionally with patch shapes.
        """
        batch_size = len(images)
        _ = self.forward_modules["feature_aggregator"].eval()

        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.make_patch(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]

        features = self.patch_maker.patch_alignment(features, patch_shapes)
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["pre_adapt_aggregator"](features)

        if provide_patch_shapes:
            return self.patch_maker.recombinant_features(features, batch_size), patch_shapes[0]
        return self.patch_maker.recombinant_features(features, batch_size)

    def dim_reduction(self, features: torch.Tensor) -> torch.Tensor:
        """Apply dimensionality reduction to features."""
        return self.forward_modules["feat_projector"](features)

    def _get_edge_mask(
        self, feat_resolution: Tuple[int, int, int], retention_rate: float = 0.94
    ) -> torch.Tensor:
        """Generate edge mask for feature maps."""
        _, h, w = feat_resolution
        mask = torch.zeros((h, w), dtype=torch.float32).to(self.device)

        rect_height, rect_width = int(h * retention_rate), int(w * retention_rate)
        top_left = ((h - rect_height) // 2, (w - rect_width) // 2)
        bottom_right = (top_left[0] + rect_height, top_left[1] + rect_width)

        mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1
        return mask.unsqueeze(0).unsqueeze(0)

    def feature_update(
        self,
        features: torch.Tensor,
        instance_scores: torch.Tensor,
        patch_scores: torch.Tensor,
    ) -> None:
        """Update global and local memory banks with new features."""
        num = 0 if self.glo_memory is None else self.glo_memory.size(0)
        _num = features.size(0)

        if num == 0:
            if _num <= self.glo_memory_num:
                self.glo_memory = features
                self.glo_scores = instance_scores
                self.gol_patch_scores = patch_scores
            else:
                memory_limit = min(3 * self.loc_memory_num, _num - self.glo_memory_num)
                scores, indices = torch.topk(
                    instance_scores, self.glo_memory_num + memory_limit, dim=0, largest=False
                )
                self.glo_scores = scores[:self.glo_memory_num]
                self.glo_memory = features[indices[:self.glo_memory_num]]
                self.gol_patch_scores = patch_scores[indices[:self.glo_memory_num]]
                self._local_feature_update(
                    features[indices[self.glo_memory_num:]],
                    patch_scores[indices[self.glo_memory_num:]],
                )
        else:
            if (num + _num) <= self.glo_memory_num:
                self.glo_memory = torch.cat([self.glo_memory, features], dim=0)
                self.glo_scores = torch.cat([self.glo_scores, instance_scores], dim=0)
                self.gol_patch_scores = torch.cat([self.gol_patch_scores, patch_scores], dim=0)
            else:
                memory_limit = min(3 * self.loc_memory_num, (num + _num) - self.glo_memory_num)
                cat_scores = torch.cat([self.glo_scores, instance_scores], dim=0)
                cat_patch_scores = torch.cat([self.gol_patch_scores, patch_scores], dim=0)
                cat_features = torch.cat([self.glo_memory, features], dim=0)
                scores, indices = torch.topk(
                    cat_scores, self.glo_memory_num + memory_limit, dim=0, largest=False
                )
                self.glo_scores = scores[:self.glo_memory_num]
                self.glo_memory = cat_features[indices[:self.glo_memory_num]]
                self.gol_patch_scores = cat_patch_scores[indices[:self.glo_memory_num]]
                self._local_feature_update(
                    cat_features[indices[self.glo_memory_num:]],
                    cat_patch_scores[indices[self.glo_memory_num:]],
                )

    def _local_feature_update(
        self, features: torch.Tensor, scores: torch.Tensor
    ) -> None:
        """Update local memory bank."""
        num = 0 if self.loc_memory is None else self.loc_memory.size(0)
        _num = features.size(0)
        
        if num == 0:
            if _num <= self.loc_memory_num:
                self.loc_memory = features
                self.loc_scores = scores
            else:
                self.loc_memory, self.loc_scores = feature_select(
                    features, scores, self.loc_memory_num
                )
        else:
            if (num + _num) <= self.loc_memory_num:
                self.loc_memory = torch.cat([self.loc_memory, features], dim=0)
                self.loc_scores = torch.cat([self.loc_scores, scores], dim=0)
            else:
                self.loc_memory, self.loc_scores = feature_select(
                    torch.cat([self.loc_memory, features], dim=0),
                    torch.cat([self.loc_scores, scores], dim=0),
                    self.loc_memory_num,
                )

    def registration(
        self, ref_img: torch.Tensor, align_img: torch.Tensor
    ) -> Tuple[torch.Tensor, AffineTrans, torch.Tensor]:
        """
        Perform image registration between reference and alignment images.
        
        Args:
            ref_img: Reference image tensor.
            align_img: Images to align to reference.
            
        Returns:
            Tuple of (registered images, affine model, transformation matrix).
        """
        matrix, model = self._batch_img_registration(
            self._resize(ref_img, 128), self._resize(align_img, 128)
        )
        
        if self.reg_msk is None:
            model.img_shape = ref_img.size(2)
            self.reg_msk = model.generate_circular_mask().to(self.device)
        
        registered, _ = model.affine_trans(
            matrix, align_img, padding_mode=self.args.padding_mode
        )
        return registered, model, matrix

    def _batch_img_registration(
        self,
        ref_img: torch.Tensor,
        align_img: torch.Tensor,
        lr_theta: float = 0.1,
        lr_trans: float = 0.005,
        num_epochs: int = 100,
        error_tolerance_factor: float = 0.6,
    ) -> Tuple[torch.Tensor, AffineTrans]:
        """Perform batch image registration optimization."""
        b, _, h, _ = align_img.shape
        num_epochs = max(int((b / 16) * num_epochs), 30)
        model = AffineTrans(batch_size=b, img_shape=h).to(self.device)

        # Prepare optimizer parameters
        optimizer_params = [{"params": model.translation, "lr": lr_trans}]
        if self.use_rotation:
            optimizer_params.append({"params": [model.theta], "lr": lr_theta})
        else:
            model.fixed_parameters(fixed_rotation=True)

        optimizer = optim.Adam(optimizer_params)
        start_matrix = model.get_affine_matrix()

        if self.use_rotation:
            # First optimization phase
            registered_1st = self._matrix_opt(ref_img, align_img, model, optimizer, num_epochs)
            matrix = model.get_affine_matrix()
            _, blank_msk_1st = model.affine_trans(matrix, align_img, padding_mode="zeros")

            # Adjust theta and perform second optimization phase
            model.theta.data.copy_(model.theta.data + math.pi)
            registered_2nd = self._matrix_opt(ref_img, align_img, model, optimizer, num_epochs)
            matrix_2nd = model.get_affine_matrix()
            _, blank_msk_2nd = model.affine_trans(matrix_2nd, align_img, padding_mode="zeros")

            # Calculate merged mask and errors
            merged_mask = (~torch.logical_or(blank_msk_1st, blank_msk_2nd)).float()
            error_1st = torch.mean(
                (registered_1st * merged_mask - ref_img * merged_mask) ** 2, dim=(1, 2, 3)
            )
            error_2nd = torch.mean(
                (registered_2nd * merged_mask - ref_img * merged_mask) ** 2, dim=(1, 2, 3)
            )

            # Determine conditions for matrix update
            error_start = torch.mean(
                (align_img * merged_mask - ref_img * merged_mask) ** 2, dim=(1, 2, 3)
            )
            condition1 = error_1st > error_2nd
            condition2 = (error_1st > (error_start * error_tolerance_factor)) & (
                error_2nd > (error_start * error_tolerance_factor)
            )
            matrix[condition1] = matrix_2nd[condition1]
        else:
            # Translation-only optimization
            registered = self._matrix_opt(ref_img, align_img, model, optimizer, num_epochs // 4)
            matrix = model.get_affine_matrix()
            _, blank_msk = model.affine_trans(matrix, align_img, padding_mode="zeros")

            error_start = torch.mean(
                (align_img * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                dim=(1, 2, 3),
            )
            error = torch.mean(
                (registered * (~blank_msk).float() - ref_img * (~blank_msk).float()) ** 2,
                dim=(1, 2, 3),
            )
            condition2 = error > (error_start * error_tolerance_factor)

        # Update matrix based on conditions
        matrix[condition2] = start_matrix[condition2]
        return matrix, model

    def _matrix_opt(
        self,
        ref_img: torch.Tensor,
        align_img: torch.Tensor,
        model: AffineTrans,
        optimizer: optim.Optimizer,
        num_epochs: int,
    ) -> torch.Tensor:
        """Optimize transformation matrix."""
        registered_img, _ = model(align_img)
        mask = model.mask.to(self.device)

        for _ in range(num_epochs):
            loss = torch.mean((ref_img * mask - registered_img * mask) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            registered_img, _ = model(align_img)

        return registered_img

    @staticmethod
    def _resize(x: torch.Tensor, target_size: int) -> torch.Tensor:
        """Resize tensor to target size."""
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    def few_shot_memory(self, normal_images: List[Dict[str, Any]]) -> None:
        """
        Build few-shot memory from normal images.
        
        Args:
            normal_images: List of dictionaries containing 'image' and 'filename' keys.
        """
        input_images = []
        print("The selected normal sample is:")
        for item in normal_images:
            input_images.append(item["image"])
            print(item["filename"])
        
        input_images = torch.stack(input_images).to(self.device)
        
        if len(input_images) != 1 and self.strategy != "texture":
            selected_index = select_reference_image_with_orb(input_images)
            indices = torch.arange(input_images.shape[0], device=self.device)
            first_img = input_images[selected_index][None]
            registered = self.registration(first_img, input_images[indices != selected_index])[0]
            input_images = torch.cat([first_img, registered.detach()], dim=0)
        
        feature_maps = self.features_extract(input_images)
        feature_maps = self.dim_reduction(feature_maps)
        self.save_knowledge(input_images[0][None], feature_maps)
        self.knowledge = self.load_knowledge()
        
        n, c, _, _ = self.knowledge["features"].shape
        self.knowledge["features"] = self.knowledge["features"].reshape(n, c, -1).unsqueeze(2)

    def scoring(self, feature_maps: torch.Tensor, nbr: int = 3) -> torch.Tensor:
        """
        Compute anomaly scores using neighborhood mutual scoring mechanism.
        
        Args:
            feature_maps: Feature maps tensor of shape (B, C, H, W).
            nbr: Neighborhood size for scoring.
            
        Returns:
            Patch-level anomaly scores.
        """
        b, c, h, w = feature_maps.shape

        def process_batches(x, y, input_n, input_nbr):
            batch_size = min(math.ceil(16 / input_nbr), input_n)
            col_scores = []
            for start in range(0, input_n, batch_size):
                end = min(start + batch_size, input_n)
                y_batch = y[start:end]
                y_batch = self.patch_maker.apply_neighbor_unfold(y_batch, input_nbr)
                scores = -F.max_pool2d(
                    -torch.norm(
                        (x.unsqueeze(-1).unsqueeze(-1) - y_batch)
                        .permute(0, 1, 2, 4, 3, 5)
                        .reshape(y_batch.shape[0], c, h * nbr, w * nbr),
                        dim=1,
                    ),
                    kernel_size=nbr,
                    stride=nbr,
                )
                col_scores.append(scores)
            return torch.cat(col_scores, dim=0)

        patch_scores = []

        for i in range(b):
            current_feature_map = feature_maps[i : i + 1]
            other_feature_map = torch.cat(
                [feature_maps[:i], feature_maps[i + 1 :]], dim=0
            )
            n = b - 1

            # Add memory features
            if self.glo_memory is not None:
                n += len(self.glo_memory)
                other_feature_map = torch.cat([other_feature_map, self.glo_memory], dim=0)
            if self.loc_memory is not None:
                n += len(self.loc_memory)
                other_feature_map = torch.cat([other_feature_map, self.loc_memory], dim=0)

            batch_scores = process_batches(current_feature_map, other_feature_map, n, nbr)

            if self.k_shot != 0:
                knowledge_scores = []
                for j in range(self.k_shot):
                    knowledge_scores.append(
                        torch.min(
                            torch.norm(
                                current_feature_map.reshape(1, c, -1).unsqueeze(-1)
                                - self.knowledge["features"][j][None],
                                dim=1,
                            ),
                            dim=-1,
                        )[0]
                    )
                knowledge_scores = torch.cat(knowledge_scores, dim=0).reshape(-1, h, w)

                patch_scores.append(
                    torch.mean(
                        torch.topk(
                            torch.cat([knowledge_scores, batch_scores], dim=0),
                            k=max(math.ceil(((n - 1) + self.k_shot) * self.args.k_min), 1),
                            dim=0,
                            largest=False,
                        )[0],
                        dim=0,
                    )
                )
            else:
                patch_scores.append(
                    torch.mean(
                        torch.topk(
                            batch_scores,
                            k=max(math.ceil((n - 1) * self.args.k_min), 1),
                            dim=0,
                            largest=False,
                        )[0],
                        dim=0,
                    )
                )

        return torch.stack(patch_scores)

    def batch_predict(
        self,
        input_images: torch.Tensor,
        matrix: Optional[torch.Tensor],
        reg_model: Optional[AffineTrans],
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Perform batch prediction on input images.
        
        Args:
            input_images: Input images tensor.
            matrix: Transformation matrix (None if no registration).
            reg_model: Registration model (None if no registration).
            
        Returns:
            Tuple of (image-level scores, pixel-level anomaly maps).
        """
        feature_maps = self.features_extract(input_images)
        feature_maps = self.dim_reduction(feature_maps)
        b, c, h, w = feature_maps.shape

        # Neighborhood Mutual Scoring Mechanism
        with torch.no_grad():
            patch_scores = self.scoring(feature_maps, nbr=self.nbr)

        # TPC (Texture-aware Prototype Calibration)
        if self.strategy != "remain":
            mean_feature_maps = batch_center_crop(
                feature_maps,
                (int(h * self.args.feat_crop_ratio), int(w * self.args.feat_crop_ratio)),
            )
        else:
            mean_feature_maps = feature_maps
        
        self_scores = torch.norm(
            feature_maps - torch.mean(mean_feature_maps, dim=(2, 3)).unsqueeze(2).unsqueeze(3),
            dim=1,
        )
        patch_scores = patch_scores * torch.min(
            torch.stack([patch_scores, self_scores], dim=1), dim=1
        )[0]

        # Inverse transformation
        if matrix is not None:
            matrix_inv = reg_model.invert_affine_matrices(matrix)
            revised_patch_scores = reg_model.affine_trans(
                matrix_inv, patch_scores.unsqueeze(1), padding_mode="zeros", need_blank_mask=False
            ).squeeze(1)
            revised_patch_scores = revised_patch_scores * self._resize(self.reg_msk, (h, w)).squeeze(1)
        else:
            revised_patch_scores = patch_scores * self._resize(self.edge_msk, (h, w)).squeeze(1)

        img_scores = torch.mean(
            torch.topk(
                revised_patch_scores.view(revised_patch_scores.size(0), -1), k=self.top_k
            ).values,
            dim=-1,
        ).detach()

        self.feature_update(feature_maps, img_scores, patch_scores)
        anomaly_maps = self.anomaly_segment.convert_to_segmentation(revised_patch_scores)

        return img_scores.cpu(), anomaly_maps

    def predict(self) -> Tuple[torch.Tensor, ...]:
        """
        Run prediction on the entire dataset.
        
        Returns:
            Tuple containing normalized scores, labels, ground truth, timing info, and images.
        """
        all_scores_image = []
        gt_labels = []
        all_scores_pixel = []
        labels = []
        images = []
        regs = []

        below_first_batch_lower_limit = self.first_batch_lower_limit > self.args.batch_size
        data_len = len(self.dataloader["dataset_set"])
        backup_data = None

        first_img = self.knowledge["images"] if self.k_shot != 0 else None
        start_time = time.time()
        is_start = True

        for idx, data in tqdm.tqdm(
            enumerate(self.dataloader["dataset_set"], 0),
            desc="Predicting...",
            leave=False,
            total=data_len,
        ):
            if below_first_batch_lower_limit and idx == 0:
                backup_data = {k: data[k] for k in ["image", "label", "mask"]}
                continue

            if below_first_batch_lower_limit:
                for key in ["image", "label", "mask"]:
                    backup_data[key] = torch.cat([backup_data[key], data[key]], dim=0)
                if self.first_batch_lower_limit > len(backup_data["image"]):
                    continue
                else:
                    input_images = backup_data["image"]
                    img_label = backup_data["label"]
                    img_mask = backup_data["mask"]
                    below_first_batch_lower_limit = False
            else:
                input_images = data["image"]
                img_label = data["label"]
                img_mask = data["mask"]

            # PMGR (Prototype-based Multi-scale Global Registration)
            if self.strategy == "remain":
                reg_images = input_images.to(self.device)
                matrix, reg_model = None, None
            else:
                if is_start and self.k_shot == 0:
                    selected_index = select_reference_image_with_orb(input_images)
                    first_img = input_images[selected_index][None].to(self.device)
                    is_start = False
                registered, reg_model, matrix = self.registration(
                    first_img, input_images.to(self.device)
                )
                reg_images = registered.detach()

            scores_image, scores_pixel = self.batch_predict(reg_images, matrix, reg_model)
            
            for scores in scores_pixel:
                all_scores_pixel.append(torch.from_numpy(scores).unsqueeze(0))
            labels.append(img_label)
            all_scores_image.append(scores_image.view(-1))
            gt_labels.append(img_mask)
            images.append(input_images)
            regs.append(reg_images.cpu())

        end_time = time.time()

        all_scores_pixel = torch.cat(all_scores_pixel, dim=0)
        labels = torch.cat(labels, dim=0)
        all_scores_image = torch.cat(all_scores_image, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        images = torch.cat(images, dim=0)
        regs = torch.cat(regs, dim=0)

        return (
            normalize_data(all_scores_image),
            normalize_data(all_scores_pixel),
            labels,
            gt_labels,
            end_time - start_time,
            images,
            regs,
        )

    def test(self) -> OrderedDict:
        """
        Run evaluation and compute metrics.
        
        Returns:
            OrderedDict containing all evaluation metrics.
        """
        (
            all_scores_image,
            all_scores_pixel,
            labels,
            gt_labels,
            test_time,
            images,
            regs,
        ) = self.predict()

        print("   Calculating metrics..")
        metrics_list = ["img_auc", "pixel_auc", "img_ap", "pixel_ap", "pro", "pixel_f1", "img_f1"]
        metrics = compute_metrics(
            labels, all_scores_image, gt_labels, all_scores_pixel, metrics_to_compute=metrics_list
        )

        output = [
            f"I-AUROC: {metrics['img_auc']:.3f}",
            f"P-AUROC: {metrics['pixel_auc']:.3f}",
            f"I-AP: {metrics['img_ap']:.3f}",
            f"P-AP: {metrics['pixel_ap']:.3f}",
            f"I-f1-max: {metrics['img_f1_max']:.3f}",
            f"P-f1-max: {metrics['pixel_f1_max']:.3f}",
            f"PRO: {metrics['pro']:.3f}",
            f"Inference time: {images.size(0) / test_time:.3f} fps",
        ]

        print("   Metrics: | " + " | ".join(output) + " |")
        print("\n\t---------\n")

        if self.args.is_plot:
            segment_path = os.path.join(
                self.args.save_path, self.args.resume, f"{self.class_name}/segment_results"
            )
            os.makedirs(segment_path, exist_ok=True)

            plot_segmentation_images(
                images.numpy(),
                regs.numpy(),
                all_scores_pixel.numpy(),
                gt_labels.numpy(),
                metrics["threshold"],
                segment_path,
            )

        return OrderedDict(
            [
                ("img_f1", metrics["img_f1_max"]),
                ("pixel_f1", metrics["pixel_f1_max"]),
                ("img_auc", metrics["img_auc"]),
                ("pixel_auc", metrics["pixel_auc"]),
                ("img_ap", metrics["img_ap"]),
                ("pixel_ap", metrics["pixel_ap"]),
                ("threshold", metrics["threshold"]),
                ("pro", metrics["pro"]),
                ("test_time", images.size(0) / test_time),
            ]
        )