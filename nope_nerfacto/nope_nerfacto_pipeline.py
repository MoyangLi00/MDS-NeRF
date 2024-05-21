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
A pipeline that dynamically chooses the number of rays to sample.
"""

from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import os
import torch
import torch.distributed as dist
import torchvision.utils as vutils
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nope_nerfacto.losses import NopeNerfDepthLossType
from nerfstudio.model_components.losses import DepthLossType
import numpy as np


@dataclass
class NopeNerfactoPipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: NopeNerfactoPipeline)
    output_path: Path = Path("./outputs/eval_images/experiment_name")
    """Path to save rendered images to."""
    use_depth_loss_step: int = 1000
    """Step to start using depth loss."""
    use_depth_loss: bool = True
    """Whether to use depth loss."""
    use_filter_outliers: bool = False
    """Whether to filter out outliers."""
    step_to_second_stage: int = 2000
    """Step to start using the second stage."""
    use_depth_warping_loss: bool = False
    """Whether to use depth warping loss."""
    Second_depth_type: NopeNerfDepthLossType = NopeNerfDepthLossType.DS_NERF
    """Second depth type."""


class NopeNerfactoPipeline(VanillaPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    config: NopeNerfactoPipelineConfig
    datamanager: VanillaDataManager

    def __init__(
        self,
        config: NopeNerfactoPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.scale_factor = self.datamanager.dataparser.config.depth_unit_scale_factor * self.datamanager.train_dataparser_outputs.dataparser_scale

    
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        if step > self.config.step_to_second_stage and self.config.use_filter_outliers:
            # self.model.config.depth_loss_type = NopeNerfDepthLossType.RELATIVE_LOSS
            # self.model.config.depth_loss_type = DepthLossType.DS_NERF
            if self.model.config.depth_loss_type == NopeNerfDepthLossType.DS_NERF:
                self.model.config.depth_loss_type = DepthLossType.DS_NERF
            else:
                self.model.config.depth_loss_type = self.config.Second_depth_type
        if (self.config.use_depth_loss) and (step > self.config.use_depth_loss_step):
            self.model.config.use_depth_loss = True

        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        if (self.config.use_depth_warping_loss) and (step < 2000):
            image_id = batch["indices"][:, 0].to(self.device)
            # TODO: test!!!
            # pred_depth = model_outputs["expected_depth"]
            # P3d_first_view = ray_bundle.origins + ray_bundle.directions * pred_depth
            # # only update the depth loss for the second view
            # P3d_first_view = P3d_first_view.detach().clone()
            # cam_vir_id = torch.randint_like(image_id, 0, self.model.field.num_images, device=self.device)

            aligned_depth = batch["depth_image"].to(self.device) * self.model.scale_networks(image_id) + self.model.offset_networks(image_id)
            P3d_first_view = ray_bundle.origins + ray_bundle.directions * aligned_depth
            cam_vir_id = torch.randint_like(image_id, 0, self.model.field.num_images, device=self.device)
            P3d_first_view = P3d_first_view.detach().clone()

            # # TODO: test!!!
            # cam_vir_id = image_id

            c2ws = self.datamanager.train_dataset.cameras.camera_to_worlds.to(self.device)[cam_vir_id]
            fx = self.datamanager.train_dataset.cameras.fx.squeeze()
            fy = self.datamanager.train_dataset.cameras.fy.squeeze()
            cx = self.datamanager.train_dataset.cameras.cx.squeeze()
            cy = self.datamanager.train_dataset.cameras.cy.squeeze()
            instrinsics = torch.eye(3, device=self.device).unsqueeze(0).repeat(len(cam_vir_id), 1, 1)
            instrinsics[:, 0, 0] = fx.to(self.device)[cam_vir_id]
            instrinsics[:, 1, 1] = fy.to(self.device)[cam_vir_id]
            instrinsics[:, 0, 2] = cx.to(self.device)[cam_vir_id]
            instrinsics[:, 1, 2] = cy.to(self.device)[cam_vir_id]
            image_height = self.datamanager.train_dataset.cameras.height[0, 0]
            image_width = self.datamanager.train_dataset.cameras.width[0, 0]

            # transform to homogeneous coordinates
            c2ws = torch.concat([c2ws, torch.tensor([0, 0, 0, 1], device=self.device).unsqueeze(0).repeat(len(cam_vir_id), 1, 1)], dim=1)
            P3d_first_view = torch.cat([P3d_first_view, torch.ones_like(P3d_first_view[:, :1])], dim=-1) # [batch, 4]

            P3d_c2 = (instrinsics @ (torch.linalg.inv(c2ws) @ P3d_first_view.unsqueeze(-1))[:,:3,:])
            P3d_c2 = (P3d_c2 / P3d_c2[:, 2:3, :]).int() # [batch, 3, 1]

            # openGL coordinate system
            P3d_c2[:, 0, 0] = image_width - P3d_c2[:, 0, 0]

            P3d_c2_mask = (P3d_c2[:, 0, 0] >= 0) & (P3d_c2[:, 0, 0] < image_width) & (P3d_c2[:, 1, 0] >= 0) & (P3d_c2[:, 1, 0] < image_height)
            # only select the valid points within the region of image
            P3d_xy = P3d_c2[:, :2, :][P3d_c2_mask]
            cam_vir_id = cam_vir_id[P3d_c2_mask]
            all_batch = next(self.datamanager.iter_train_image_dataloader)
            aligned_depth_vir = all_batch["depth_image"].to(self.device)[cam_vir_id, P3d_xy[:,1, 0], P3d_xy[:, 0, 0]] * self.model.scale_networks(cam_vir_id) + self.model.offset_networks(cam_vir_id)
            measure_depth =  (P3d_first_view[:,:3] - c2ws[:, :3, 3])[P3d_c2_mask].norm(dim=-1)
            loss_dict["depth_warping_loss"] = torch.nn.functional.mse_loss(measure_depth.squeeze(), aligned_depth_vir.squeeze())

        return model_outputs, loss_dict, metrics_dict
    

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        output_path = self.config.output_path / f"{step:07d}"
        os.makedirs(output_path, exist_ok=True)
        
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            idx = 0
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in images_dict.keys():
                        image = images_dict[key]  # [H, W, C] order
                        vutils.save_image(image.permute(2, 0, 1).cpu(), output_path / f"eval_{key}_{idx:04d}.png")
                
                depth_vanilla_value = (outputs["depth"] / self.scale_factor).squeeze()
                np.save(output_path / f"depth_{idx:05d}.npy", depth_vanilla_value.cpu().numpy())

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx += 1
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()

        scales, offsets = self.model.get_scales_and_offsets()

        np.save(output_path / "scales.npy", scales.cpu().numpy())
        np.save(output_path / "offsets.npy", offsets.cpu().numpy()/self.scale_factor)
        
        # write scales and offsets to file
        with open(output_path / "scales.txt", "w") as f:
            f.write(str(scales))
        with open(output_path / "offsets.txt", "w") as f:
            f.write(str(offsets))

        return metrics_dict