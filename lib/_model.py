import tqdm
import time
import math
from torch import optim
from collections import OrderedDict
from lib.utils import os, torch, np, plt, denormalization, plot_segmentation_images_for_paper
from lib.utils import normalize_data, plot_segmentation_images, compute_metrics, batch_center_crop, \
    select_features, select_reference_image, select_reference_image_with_orb
from lib.common import F
from lib.common import PatchMaker, NetworkFeatureAggregator, Preprocessing, Aggregator, RescaleSegment, \
    FeatureMapProjection, AffineTrans


# from tool.visualization import show_batch_images


class DyNorm:
    """
    CaAD: zero-shot industrial image anomaly detection via conformity analysis.
    DyNorm: zero-shot industrial image anomaly detection without prompts.
    """

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
            first_batch_lower_limit=4
    ):
        self.class_name = dataloader['cls_name']
        print(f'Class name: {self.class_name}[{strategy}]')
        self.args = args
        self.strategy = strategy
        self.first_batch_lower_limit = first_batch_lower_limit
        self.knwldg_dir = os.path.join(args.save_path, args.resume, f'{self.class_name}/prior_knowledge')
        self.backbone = backbone.to(device)

        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.k_shot = args.k_shot
        self.dataloader = dataloader
        self.top_k = top_k
        self.nbr = args.nbr

        self.patch_maker = PatchMaker(patch_size, stride=patch_stride)
        self.use_rotation = True if strategy == 'transform' else False

        if self.k_shot != 0:
            self.knowledge = self.load_knowledge()
            if self.knowledge is None:
                print('Please first use the few_shot_memory method to create knowledge.')
            else:
                n, c, _, _ = self.knowledge['features'].shape
                self.knowledge['features'] = self.knowledge['features'].reshape(n, c, -1).unsqueeze(2)
        else:
            self.knowledge: torch.Tensor = None

        self.object_msk: torch.Tensor = None
        self.texture_msk = self.get_texture_mask(input_shape)

        self.total_recall: torch.Tensor = None
        self.total_scores: torch.Tensor = None
        self.total_num = args.cns_num
        self.memory: torch.Tensor = None
        self.memory_optimizer: optim.Optimizer = None

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension

        pre_adapt_aggregator = Aggregator(
            target_dim=target_embed_dimension
        )

        self.forward_modules["pre_adapt_aggregator"] = pre_adapt_aggregator.to(self.device)

        self.anomaly_segment = RescaleSegment(
            device=self.device, target_size=input_shape[-2:]
        )
        feat_projector = FeatureMapProjection(target_embed_dimension, proj_dim)
        self.forward_modules["feat_projector"] = feat_projector.to(self.device)

    def save_knowledge(self, images, features):
        os.makedirs(self.knwldg_dir, exist_ok=True)
        torch.save({'images': images, 'features': features},
                   '%s/knowledge.pth' % self.knwldg_dir)

    def load_knowledge(self):
        """
        Load the saved knowledge from the knowledge directory.

        Returns:
            dict: A dictionary containing the saved knowledge, with keys 'image', 'features', and 'msk'.
        """
        knowledge_path = os.path.join(self.knwldg_dir, 'knowledge.pth')
        if os.path.exists(knowledge_path):
            knowledge = torch.load(knowledge_path, map_location=self.device)
            return knowledge
        else:
            return None

    def features_extract(self, images, provide_patch_shapes=False):
        """Extract patch features."""

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

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["pre_adapt_aggregator"](features)

        if provide_patch_shapes:
            return self.patch_maker.recombinant_features(features, batch_size), patch_shapes[0]

        return self.patch_maker.recombinant_features(features, batch_size)

    def dim_reduction(self, features):
        return self.forward_modules["feat_projector"](features)

    def get_texture_mask(self, feat_resolution, retention_rate=0.94):
        _, h, w = feat_resolution
        mask = torch.zeros((h, w), dtype=torch.float32).to(self.device)

        rect_height, rect_width = int(h * retention_rate), int(w * retention_rate)

        top_left = ((h - rect_height) // 2, (w - rect_width) // 2)
        bottom_right = (top_left[0] + rect_height, top_left[1] + rect_width)

        mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1

        return mask.unsqueeze(0).unsqueeze(0)

    def memory_update(self, memory_scores, scores):
        loss = torch.mean((memory_scores - scores) ** 2)
        self.memory_optimizer.zero_grad()
        loss.backward()
        self.memory_optimizer.step()

        return loss  # Optionally return the loss for monitoring

    def feature_update(self, _features, _scores):
        num = 0 if self.total_recall is None else self.total_recall.size(0)
        _num = _features.size(0)
        if num == 0:
            if _num <= self.total_num:
                self.total_recall = _features
                self.total_scores = _scores
            else:
                self.total_scores, indices = torch.topk(_scores, self.total_num, dim=0, largest=False)
                self.total_recall = _features[indices]
        else:
            if (num + _num) <= self.total_num:
                self.total_recall = torch.cat([self.total_recall, _features], dim=0)
                self.total_scores = torch.cat([self.total_scores, _scores], dim=0)
            else:
                self.total_scores, indices = torch.topk(torch.cat([self.total_scores, _scores], dim=0),
                                                        self.total_num, dim=0, largest=False)
                self.total_recall = torch.cat([self.total_recall, _features], dim=0)[indices]

    def registration(self, ref_img, align_img):

        matrix, model = self.batch_img_registration(
            self.re_size(ref_img, 128), self.re_size(align_img, 128)
        )
        # matrix, model = self.batch_img_registration(ref_img, align_img)
        if self.object_msk is None:
            model.img_shape = ref_img.size(2)
            self.object_msk = model.generate_circular_mask().to(self.device)
        registered, _ = model.affine_trans(matrix, align_img, padding_mode=self.args.padding_mode)

        return registered, model, matrix

    def batch_img_registration(self, ref_img, align_img, lr_theta=0.1, lr_trans=0.005,
                               num_epochs=100, error_tolerance_factor=0.6):
        b, _, h, _ = align_img.shape
        num_epochs = max(int((b / 16) * num_epochs), 30)
        model = AffineTrans(batch_size=b, img_shape=h, device=self.device).to(self.device)
        # if self.object_msk is None:
        #     self.object_msk = model.generate_circular_mask()

        # Prepare optimizer parameters
        optimizer_params = [{'params': model.translation, 'lr': lr_trans}]
        if self.use_rotation:
            optimizer_params.append({'params': [model.theta], 'lr': lr_theta})
        else:
            model.fixed_parameters(fixed_rotation=True)

        optimizer = optim.Adam(optimizer_params)
        start_matrix = model.get_affine_matrix()

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

        # Update matrix based on conditions
        matrix[condition2] = start_matrix[condition2]

        return matrix, model

    def matrix_opt(self, ref_img, align_img, model, optimizer, num_epochs):
        registered_img, _ = model(align_img)
        mask = model.mask.to(self.device)

        for epoch in range(num_epochs):
            loss = torch.mean((ref_img * mask - registered_img * mask) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            registered_img, _ = model(align_img)

        return registered_img

    @staticmethod
    def re_size(x, target_size):
        x = F.interpolate(
            x, size=target_size, mode="bilinear", align_corners=False
        )

        return x

    def few_shot_memory(self, normal_images):
        input_images = []
        print('The selected normal sample is:')
        for i in range(len(normal_images)):
            input_images.append(normal_images[i]['image'])
            print(normal_images[i]['filename'])
        input_images = torch.stack(input_images).to(self.device)
        if len(input_images) != 1 and self.strategy != 'texture':
            selected_index = select_reference_image_with_orb(input_images)
            indices = torch.arange(input_images.shape[0], device=self.device)
            first_img = input_images[selected_index][None]
            registered = self.registration(first_img, input_images[
                indices != selected_index])[0]
            input_images = torch.cat([first_img, registered.detach()], dim=0)
        feature_maps = self.features_extract(input_images)
        feature_maps = self.dim_reduction(feature_maps)
        self.save_knowledge(input_images[0][None], feature_maps)
        self.knowledge = self.load_knowledge()
        n, c, _, _ = self.knowledge['features'].shape
        self.knowledge['features'] = self.knowledge['features'].reshape(n, c, -1).unsqueeze(2)

    def scoring(self, feature_maps, nbr=3):
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
                        (x.unsqueeze(-1).unsqueeze(-1) - y_batch
                         ).permute(0, 1, 2, 4, 3, 5).reshape(y_batch.shape[0], c, h * nbr, w * nbr), dim=1
                    ), kernel_size=nbr, stride=nbr)
                col_scores.append(scores)

            return torch.cat(col_scores, dim=0)

        patch_scores = []

        for i in range(b):
            current_feature_map = feature_maps[i:i + 1]  # (1, c, h, w)
            other_feature_map = torch.cat([feature_maps[:i], feature_maps[i + 1:]], dim=0)  # (b-1, c, h, w)

            n = b - 1

            # CNS
            if self.total_recall is not None:
                n += len(self.total_recall)
                other_feature_map = torch.cat([other_feature_map, self.total_recall], dim=0)

            if self.memory is not None:
                n += len(self.memory)
                other_feature_map = torch.cat([other_feature_map, self.memory], dim=0)

            # original code
            # # (n, c, h, w) -> (n, c, h, w, nbr, nbr)
            # other_feature_map = self.patch_maker.apply_neighbor_unfold(other_feature_map, nbr)
            #
            # # calculate distance
            # # (n, c, h, w, nbr, nbr) -> (n, c, h*nbr, w*hbr) -> (n, h*nbr, w*nbr) -> (n, h, w) -> (h, w)
            # batch_scores = -F.max_pool2d(
            #     -torch.norm(
            #         (current_feature_map.unsqueeze(-1).unsqueeze(-1) - other_feature_map
            #          ).permute(0, 1, 2, 4, 3, 5).reshape(n, c, h * nbr, w * nbr), dim=1
            #     ), kernel_size=nbr, stride=nbr)

            # speed optimize
            batch_scores = process_batches(current_feature_map, other_feature_map, n, nbr)

            if self.k_shot != 0:
                knowledge_sores = []
                for j in range(0, self.k_shot):
                    knowledge_sores.append(torch.min(
                        torch.norm(
                            current_feature_map.reshape(1, c, -1).unsqueeze(-1) - self.knowledge['features'][j][None],
                            dim=1), dim=-1
                    )[0])
                knowledge_sores = torch.cat(knowledge_sores, dim=0).reshape(-1, h, w)

                patch_scores.append(
                    torch.mean(
                        torch.topk(
                            torch.cat([knowledge_sores, batch_scores], dim=0),
                            k=max(math.ceil((n - 1 + self.k_shot) * self.args.k_min), 1), dim=0, largest=False
                        )[0], dim=0
                    )
                )
            else:
                patch_scores.append(
                    torch.mean(
                        torch.topk(batch_scores, k=max(math.ceil((n - 1) * self.args.k_min), 1), dim=0, largest=False
                                   )[0], dim=0)
                )

        patch_scores = torch.stack(patch_scores)

        return patch_scores

    def batch_predict(self, input_images, matrix, reg_model, is_start=True):
        feature_maps = self.features_extract(input_images)
        feature_maps = self.dim_reduction(feature_maps)
        b, c, h, w = feature_maps.shape

        # Neighborhood Mutual Scoring
        with torch.no_grad():
            patch_scores = self.scoring(feature_maps, nbr=self.nbr)

        # # FNL
        # if self.args.memory_num > 0:
        #     if is_start:  # Memory bank initialization
        #         # self.memory = select_features(features=feature_maps, scores=patch_scores, n=self.args.memory_num)
        #         self.memory = feature_maps[:3].clone()
        #         self.memory = torch.nn.Parameter(self.memory)
        #         self.memory_optimizer = torch.optim.Adam([self.memory], lr=self.args.memory_lr)
        #         for i in range(self.args.memory_freq):
        #             diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
        #             memory_scores, _ = torch.min(torch.norm(diff, dim=2), dim=1)
        #             self.memory_update(memory_scores, patch_scores)
        #     else:  # Memory update
        #         for i in range(self.args.memory_freq):
        #             diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
        #             memory_scores, _ = torch.min(torch.norm(diff, dim=2), dim=1)
        #             self.memory_update(memory_scores, patch_scores)
        # Memory bank initialization
        if self.args.memory_num > 0 and is_start:
            self.memory = select_features(features=feature_maps, scores=patch_scores, n=self.args.memory_num)
            # self.memory = feature_maps[:3].clone()
            self.memory = torch.nn.Parameter(self.memory)
            self.memory_optimizer = torch.optim.Adam([self.memory], lr=self.args.memory_lr)

        # Memory update
        for i in range(self.args.memory_freq):
            diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
            memory_scores, _ = torch.min(torch.norm(diff, dim=2), dim=1)
            self.memory_update(memory_scores, patch_scores)

        # # Memory update
        # for i in range(self.args.memory_freq):
        #     for f_i, f in enumerate(feature_maps):
        #         memory_scores = -F.max_pool2d(
        #             -torch.norm(
        #                 (
        #                         f[None, None, :, :, :, None, None] - self.patch_maker.apply_neighbor_unfold(self.memory, self.nbr)[None, :, :, :, :, :, :]
        #                 ).permute(0, 1, 2, 3, 5, 4, 6).reshape(1, self.args.memory_num, c, h * self.nbr, w * self.nbr), dim=2
        #             ), kernel_size=self.nbr, stride=self.nbr
        #         )
        #         memory_scores , _ = torch.min(memory_scores, dim=1)
        #         # diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
        #         # memory_scores, _ = torch.min(torch.norm(diff, dim=2), dim=1)
        #         self.memory_update(memory_scores, patch_scores[f_i][None])

        # # FNL
        # if is_start:
        #     self.memory = select_features(features=feature_maps, scores=patch_scores, n=self.args.memory_num)
        #     self.memory = torch.nn.Parameter(self.memory)
        #     self.memory_optimizer = torch.optim.Adam([self.memory], lr=self.args.memory_lr)
        # else:
        #     # (B, 1, C, H, W) - (1, M, C, H, W) --> (B, M, C, H, W)
        #     diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
        #     # (B, M, C, H, W) --> (B, M, H, W) --> (B, H, W)
        #     memory_scores = torch.min(torch.norm(diff, dim=2), dim=1)[0]
        #
        #     # Use torch.no_grad() to prevent gradient calculation for patch_scores
        #     with torch.no_grad():
        #         patch_scores = torch.min(torch.stack([patch_scores, memory_scores], dim=1), dim=1)[0]
        #
        #     # Memory update
        #     for i in range(self.args.memory_freq):
        #         if i > 0:
        #             diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
        #             memory_scores, _ = torch.min(torch.norm(diff, dim=2), dim=1)
        #         self.memory_update(memory_scores, patch_scores)

        # TPC
        if self.strategy != 'remain':
            mean_feature_maps = batch_center_crop(
                feature_maps, (int(h * self.args.feat_crop_ratio), int(w * self.args.feat_crop_ratio))
            )
        else:
            mean_feature_maps = feature_maps
        self_scores = torch.norm(
            feature_maps - torch.mean(mean_feature_maps, dim=(2, 3)).unsqueeze(2).unsqueeze(3), dim=1
        )
        patch_scores = (patch_scores * torch.min(torch.stack([patch_scores, self_scores], dim=1), dim=1)[0])
        # # Memory update
        # for i in range(self.args.memory_freq):
        #     diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
        #     memory_scores, _ = torch.min(torch.norm(diff, dim=2), dim=1)
        #     self.memory_update(memory_scores, torch.sqrt(patch_scores))

        # # Memory update
        # for i in range(self.args.memory_freq):
        #     for f_i, f in enumerate(feature_maps):
        #         memory_scores = -F.max_pool2d(
        #             -torch.norm(
        #                 (
        #                         f[None, None, :, :, :, None, None] - self.patch_maker.apply_neighbor_unfold(self.memory, self.nbr)[None, :, :, :, :, :, :]
        #                 ).permute(0, 1, 2, 3, 5, 4, 6).reshape(1, self.args.memory_num, c, h * self.nbr, w * self.nbr), dim=2
        #             ), kernel_size=self.nbr, stride=self.nbr
        #         )
        #         memory_scores , _ = torch.min(memory_scores, dim=1)
        #         # diff = feature_maps[:, None, :, :, :] - self.memory[None, :, :, :, :]
        #         # memory_scores, _ = torch.min(torch.norm(diff, dim=2), dim=1)
        #         self.memory_update(memory_scores, patch_scores[f_i][None])

        # Inverse
        patch_scores = patch_scores.unsqueeze(1)
        if matrix is not None:
            matrix_inv = reg_model.invert_affine_matrices(matrix)
            revised_patch_scores, _ = reg_model.affine_trans(matrix_inv, patch_scores, padding_mode='zeros')
            revised_patch_scores = revised_patch_scores * self.re_size(self.object_msk, (h, w)).squeeze(1)
        else:
            revised_patch_scores = patch_scores * self.re_size(self.texture_msk, (h, w)).squeeze(1)

        img_scores = torch.mean(torch.topk(revised_patch_scores.view(revised_patch_scores.size(0), -1),
                                           k=self.top_k).values, dim=-1).detach()
        # revised_patch_scores = torch.clamp(
        #     revised_patch_scores, max=img_scores.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # )
        self.feature_update(feature_maps, img_scores)

        anomaly_maps = self.anomaly_segment.convert_to_segmentation(revised_patch_scores.squeeze(1))

        return img_scores.cpu(), anomaly_maps

    def predict(self):
        all_scores_image = []
        gt_labels = []
        all_scores_pixel = []
        labels = []
        images = []
        regs = []

        below_first_batch_lower_limit = True if self.first_batch_lower_limit > self.args.batch_size else False
        data_len = len(self.dataloader['dataset_set'])
        backup_data = None

        first_img = self.knowledge['images'] if self.k_shot != 0 else None
        start_time = time.time()
        is_start = True
        for idx, data in tqdm.tqdm(enumerate(self.dataloader['dataset_set'], 0), desc="Predicting...", leave=False,
                                   total=data_len):
            if below_first_batch_lower_limit and idx == 0:
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
            # reg_images = input_images.to(self.device)
            # matrix, reg_model = None, None

            scores_image, scores_pixel = self.batch_predict(reg_images, matrix, reg_model, is_start=is_start)
            is_start = False
            for scores in scores_pixel:
                all_scores_pixel.append(torch.from_numpy(scores).unsqueeze(0))
            labels.append(img_label)
            all_scores_image.append(scores_image.view(-1))
            gt_labels.append(img_mask)
            images.append(input_images)
            regs.append(reg_images.cpu())
            # torch.cuda.empty_cache()

        end_time = time.time()
        all_scores_pixel = torch.cat(all_scores_pixel, dim=0)
        labels = torch.cat(labels, dim=0)
        all_scores_image = torch.cat(all_scores_image, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        images = torch.cat(images, dim=0)

        regs = torch.cat(regs, dim=0)

        return normalize_data(all_scores_image), normalize_data(
            all_scores_pixel), labels, gt_labels, end_time - start_time, images, regs

    def test(self):
        all_scores_image, all_scores_pixel, labels, gt_labels, test_time, images, regs = self.predict()
        print('   Calculating metrics..')
        metrics_list = ['img_auc', 'pixel_auc', 'img_ap', 'pixel_ap', 'pro', 'pixel_f1', 'img_f1']
        metrics = compute_metrics(
            labels, all_scores_image, gt_labels, all_scores_pixel, metrics_to_compute=metrics_list
        )
        output = [
            'I-AUROC: %.3f' % metrics['img_auc'],
            'P-AUROC: %.3f' % metrics['pixel_auc'],
            'I-AP: %.3f' % metrics['img_ap'],
            'P-AP: %.3f' % metrics['pixel_ap'],
            'I-f1-max: %.3f' % metrics['img_f1_max'],
            'P-f1-max: %.3f' % metrics['pixel_f1_max'],
            'PRO: %.3f' % metrics['pro'],
            'Inference time: %.3f fps' % (images.size(0) / test_time)
        ]

        print('   Metrics: | ' + ' | '.join(output) + ' |')
        print('\n\t---------\n')
        if self.args.is_plot:
            segment_path = os.path.join(self.args.save_path, self.args.resume, f'{self.class_name}/segment_results')
            os.makedirs(segment_path, exist_ok=True)
            res_path = os.path.join(self.args.save_path, self.args.resume, self.class_name, 'res_file.npz')
            np.savez(res_path,
                     image=images.numpy(),
                     registered=regs.numpy(),
                     score=all_scores_pixel.numpy(),
                     mask=gt_labels.numpy(),
                     thrd=metrics['threshold'])
            plot_segmentation_images(images.numpy(), all_scores_pixel.numpy(), gt_labels.numpy(),
                                     metrics['threshold'], segment_path, self.class_name)

            # plot_segmentation_images_for_paper(images.numpy(), regs.numpy(), all_scores_pixel.numpy(),
            #                                    gt_labels.numpy(), metrics['threshold'], segment_path)
        res = OrderedDict(
            [('img_f1', metrics['img_f1_max']), ('pixel_f1', metrics['pixel_f1_max']), ('img_auc', metrics['img_auc']),
             ('pixel_auc', metrics['pixel_auc']), ('img_ap', metrics['img_ap']), ('pixel_ap', metrics['pixel_ap']),
             ('threshold', metrics['threshold']), ('pro', metrics['pro']), ('test_time', images.size(0) / test_time)]
        )

        return res


if __name__ == '__main__':
    import torch.profiler
    import lib
    import yaml
    from easydict import EasyDict
    import argparse
    import warnings
    from datasets import *
    import copy
    import pandas as pd

    warnings.filterwarnings('ignore')
    path_set = "./"


    def test_one():
        conf = argparse.Namespace(save_path=f'{path_set}results')
        conf.resume = 'test-1'
        conf.layers_to_extract_from = ['layer2', 'layer3']
        conf.crop_size = 336
        conf.batch_size = 16
        conf.k_min = 0.05
        conf.cns_num = 12
        conf.feat_dim = 1024
        conf.is_plot = False
        conf.padding_mode = 'border'
        conf.k_shot = 0
        conf.memory_num = 3
        conf.memory_freq = 2
        conf.memory_lr = 0.0001
        print(conf.cns_num)
        print(conf.memory_num)
        print(conf.memory_freq)
        print(conf.memory_lr)
        conf.feat_crop_ratio = 0.9
        conf.nbr = 9
        conf.backbone = 'wideresnet50'
        # conf.backbone = 'wideresnet101'
        # conf.backbone = 'resnet50'
        choose = "mvtec"
        # choose = "visa"
        # choose = "ci"

        lib.utils.fix_seeds(0)
        _device = lib.set_torch_device([0])

        metrics = {}
        if choose == 'mvtec':
            with open(f"{path_set}datasets/config_dataset.yaml") as f:
                config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            dataset_class_names = CLASS_NAMES
            # choose_strategy = {
            #     'screw': 'transform',
            #     'metal_nut': 'transform',
            #     **{name: 'remain' if name in TEXTURE_CLASS_NAMES else 'translate' for name in CLASS_NAMES if
            #        name not in ['screw', 'metal_nut']}
            # }
            choose_strategy = {
                'screw': 'transform',
                'metal_nut': 'transform',
                **{name: 'remain' for name in CLASS_NAMES if
                   name not in ['screw', 'metal_nut']}
            }
        elif choose == 'visa':
            with open(f"{path_set}datasets/config_visa_dataset.yaml") as f:
                config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            dataset_class_names = VISA_CLASS_NAMES
            conf.resume = 'test_Z_FSAD_visa'
            choose_strategy = {
                **{name: 'remain' for name in VISA_CLASS_NAMES}
            }
            # choose_strategy = {
            #     'pcb1': 'transform',
            #     'fryum': 'transform',
            #     'cashew': 'transform',
            #     **{name: 'translate' for name in VISA_CLASS_NAMES if
            #        name not in ['pcb1', 'fryum', 'cashew']}
            # }

        elif choose == 'ci':
            with open(f"{path_set}datasets/config_ci_dataset.yaml") as f:
                config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            dataset_class_names = CI_CLASS_NAMES
            conf.resume = 'test_ci'
            choose_strategy = {
                **{name: 'remain' for name in CI_CLASS_NAMES}
            }
        else:
            with open(f"{path_set}datasets/config_bt_dataset.yaml") as f:
                config_dataset = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            dataset_class_names = BT_CLASS_NAMES
            conf.resume = 'test_Z_FSAD_bt'
            choose_strategy = {
                **{name: 'remain' for name in BT_CLASS_NAMES}
            }

        # config_dataset.dataset['input_size'] = [400, 400]
        config_dataset.dataset["batch_size"] = conf.batch_size
        cfg_dataset = config_dataset.dataset
        train_dataset = copy.deepcopy(cfg_dataset)
        cfg_dataset.update(cfg_dataset.get("test", None))
        train_dataset.update(train_dataset.get("train", None))

        for class_name in dataset_class_names:
            class_name = 'screw'
            print(class_name)
            if choose == 'mvtec':
                cls_dataloader = build_custom_dataloader(cfg_dataset, False, False, class_name=class_name)
            else:
                cls_dataloader = build_generic_dataloader(cfg_dataset, False, False, class_name=class_name)

            info_set = {"dataset_set": cls_dataloader, "cls_name": class_name}
            model = DyNorm(conf,
                           lib.load(conf.backbone),
                           conf.layers_to_extract_from,
                           info_set,
                           _device,
                           (3, conf.crop_size, conf.crop_size),
                           conf.feat_dim, conf.feat_dim, 3, 1, strategy=choose_strategy[class_name],
                           proj_dim=256
                           )
            if conf.k_shot > 0:
                train_data = select_training_data(train_dataset, k_shot=conf.k_shot, class_name=class_name)
                model.few_shot_memory(train_data)

            metrics[class_name] = model.test()
            allocated_memory = torch.cuda.memory_allocated(_device)
            reserved_memory = torch.cuda.memory_reserved(_device)
            allocated_memory_gb = allocated_memory / (1024 ** 3)
            reserved_memory_gb = reserved_memory / (1024 ** 3)
            print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")
            print(f"Reserved Memory: {reserved_memory_gb:.2f} GB")
            torch.cuda.empty_cache()

        save_dir = os.path.join(conf.save_path, conf.resume)
        os.makedirs(save_dir, exist_ok=True)
        lib.save_metrics_to_csv(save_dir, metrics)
        print(pd.read_csv(os.path.join(save_dir, "metrics.csv")))


    test_one()
