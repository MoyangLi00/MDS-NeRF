# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Nerfacto augmented with depth supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type, List

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.field_components.embedding import Embedding

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components import losses
from nerfstudio.model_components.losses import DepthLossType, depth_loss, depth_ranking_loss
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig, DepthNerfactoModel
from nerfstudio.utils import colormaps

from nope_nerfacto.losses import NopeNerfDepthLossType


@dataclass
class NopeNerfactoModelConfig(DepthNerfactoModelConfig):
    """Additional parameters for depth supervision."""

    _target: Type = field(default_factory=lambda: NopeNerfactoModel)
    depth_loss_mult: float = 1e-1
    """Lambda of the depth loss."""
    use_depth_loss: bool = False
    """Whether to use depth loss."""
    scale_init: float = 1.0
    """Initial value of the scale."""
    offset_init: float = 0.0
    """Initial value of the offset."""
    optim_scale_and_offset: bool = False
    """whether to optimize"""


class NopeNerfactoModel(DepthNerfactoModel):
    """Depth loss augmented nerfacto model.

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: DepthNerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.scale_networks = Embedding(self.field.num_images, 1)
        self.offset_networks = Embedding(self.field.num_images, 1)
        # fern
        self.scale_networks.embedding.weight.data.fill_(self.config.scale_init)
        self.offset_networks.embedding.weight.data.fill_(self.config.offset_init)
        # flower
        # self.scale_networks.embedding.weight.data.fill_(2.0)
        # self.offset_networks.embedding.weight.data.fill_(0.5)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # param_groups = super().get_param_groups()
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.optim_scale_and_offset:
            param_groups["scale_networks"] = list(self.scale_networks.parameters())
            param_groups["offset_networks"] = list(self.offset_networks.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = super().get_outputs(ray_bundle)
        
        return outputs

    def get_metrics_dict(self, outputs, batch):
        # explicitly call Nerfacto.method
        metrics_dict = NerfactoModel.get_metrics_dict(self, outputs, batch)
        if self.training and self.config.use_depth_loss:
            if (
                losses.FORCE_PSEUDODEPTH_LOSS
                and self.config.depth_loss_type not in losses.PSEUDODEPTH_COMPATIBLE_LOSSES
            ):
                raise ValueError(
                    f"Forcing pseudodepth loss, but depth loss type ({self.config.depth_loss_type}) must be one of {losses.PSEUDODEPTH_COMPATIBLE_LOSSES}"
                )
            if self.config.depth_loss_type in (DepthLossType.DS_NERF, DepthLossType.URF):
                metrics_dict["depth_loss"] = 0.0
                sigma = self._get_sigma().to(self.device)
                termination_depth = batch["depth_image"].to(self.device)
                # transform termination depth
                img_id = batch["indices"][:, 0].to(self.device)
                termination_depth = termination_depth * self.scale_networks(img_id) + self.offset_networks(img_id)
                for i in range(len(outputs["weights_list"])):
                    metrics_dict["depth_loss"] += depth_loss(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=termination_depth,
                        predicted_depth=outputs["expected_depth"],
                        sigma=sigma,
                        directions_norm=outputs["directions_norm"],
                        is_euclidean=self.config.is_euclidean_depth,
                        depth_loss_type=self.config.depth_loss_type,
                    ) / len(outputs["weights_list"])
            elif self.config.depth_loss_type in (DepthLossType.SPARSENERF_RANKING,):
                metrics_dict["depth_ranking"] = depth_ranking_loss(
                    outputs["expected_depth"], batch["depth_image"].to(self.device)
                )
            elif self.config.depth_loss_type in (NopeNerfDepthLossType.NOPE_NERF,):
                img_id = batch["indices"][:, 0].to(self.device)
                # metrics_dict["depth_loss"] = torch.nn.functional.mse_loss(
                #     outputs["expected_depth"], batch["depth_image"].to(self.device) * self.scale_networks(img_id)**2 + self.offset_networks(img_id)
                # )
                # outputs["expected_depth"] = outputs["expected_depth"].detach().clone()
                outputs["expected_depth"] = outputs["expected_depth"]
                aligned_depth = batch["depth_image"].to(self.device) * self.scale_networks(img_id) + self.offset_networks(img_id)
                nearest_depth = 0.2
                farthest_depth = 50.0
                aligned_depth = torch.clamp(aligned_depth, nearest_depth, farthest_depth)
                metrics_dict["depth_loss"] = 1e0 * torch.nn.functional.mse_loss(
                    outputs["expected_depth"], aligned_depth
                )
            elif self.config.depth_loss_type in (NopeNerfDepthLossType.RELATIVE_LOSS,):
                img_id = batch["indices"][:, 0].to(self.device)
                aligned_depth = batch["depth_image"].to(self.device) * self.scale_networks(img_id) + self.offset_networks(img_id)
                nearest_depth = 0.2
                farthest_depth = 50.0
                inlier_ratio = 0.9
                aligned_depth = torch.clamp(aligned_depth, nearest_depth, farthest_depth)
                # selct top 90% least relative error
                select_mask = ((outputs["expected_depth"] - aligned_depth) / aligned_depth).abs() < torch.quantile(
                    ((outputs["expected_depth"] - aligned_depth) / aligned_depth).abs(), inlier_ratio
                )
                metrics_dict["depth_loss"] = 1e1 * (((outputs["expected_depth"] - aligned_depth) / aligned_depth)[select_mask]**2).mean().sqrt()

            else:
                raise NotImplementedError(f"Unknown depth loss type {self.config.depth_loss_type}")

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = NerfactoModel.get_loss_dict(self, outputs, batch, metrics_dict)
        if self.training and self.config.use_depth_loss:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Appends ground truth depth to the depth image."""
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        ground_truth_depth = batch["depth_image"].to(self.device)
        if not self.config.is_euclidean_depth:
            ground_truth_depth = ground_truth_depth * outputs["directions_norm"]

        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        # predicted_depth_colormap = colormaps.apply_depth_colormap(
        #     outputs["depth"],
        #     accumulation=outputs["accumulation"],
        #     near_plane=float(torch.min(ground_truth_depth).cpu()),
        #     far_plane=float(torch.max(ground_truth_depth).cpu()),
        # )
        # do not use gt_depth for visualization
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["expected_depth"],
            accumulation=outputs["accumulation"],
        )
        images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)
        depth_mask = ground_truth_depth > 0
        metrics["depth_mse"] = float(
            torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
        )
        return metrics, images
    
    def get_scales_and_offsets(self):
        return self.scale_networks.embedding.weight.data, self.offset_networks.embedding.weight.data


