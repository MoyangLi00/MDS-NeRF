"""
NeRFIR configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nope_nerfacto.nope_nerfacto import NopeNerfactoModelConfig, NerfactoModel
from nerfstudio.pipelines.dynamic_batch import VanillaPipelineConfig
from nope_nerfacto.nope_nerfacto_pipeline import NopeNerfactoPipelineConfig

from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
from pathlib import Path
from nope_nerfacto.nope_nerfacto import NopeNerfDepthLossType
from nerfstudio.model_components.losses import DepthLossType

nope_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="nope-nerfacto",
        steps_per_eval_batch=10001,
        steps_per_eval_all_images=10001,
        steps_per_save=2000,
        # max_num_iterations=30001,
        max_num_iterations=10001,
        mixed_precision=True,
        # data=Path("./data/nerf_llff_data/room"),
        data=Path("data/Record3D/record3d_sheep"),
        # data=Path("data/DTU/DTU_scan24"),
        pipeline=NopeNerfactoPipelineConfig(
            # use_depth_loss=True,
            use_depth_loss=False,
            use_depth_loss_step=0,
            # use_filter_outliers=False,
            use_filter_outliers=True,
            # use_depth_warping_loss=True,
            use_depth_warping_loss=False,
            # step_to_second_stage=5000,
            step_to_second_stage=0,
            # Second_depth_type=NopeNerfDepthLossType.DS_NERF,
            # Second_depth_type=DepthLossType.DS_NERF,
            Second_depth_type=NopeNerfDepthLossType.RELATIVE_LOSS,
            # output_path=Path("./outputs/eval_images/Record3D/wo_Marigold_Depth/record3d_sheep/27views"),
            # output_path=Path("./outputs/eval_images/Record3D/w_Marigold_Depth/record3d_sheep/9views_Marigold_depth"),
            # output_path=Path("./outputs/eval_images/Record3D/w_Marigold_Depth/record3d_sheep/27views_GT_depth"),

            output_path=Path("./outputs/eval_images/Record3D/w_Marigold_Depth/record3d_sheep/9views_mesh"),
            # output_path=Path("./outputs/eval_images/Record3D/w_Marigold_Depth/record3d_sheep/9views_Marigold_depth_mesh"),

            # output_path=Path("./outputs/eval_images/wo_Marigold_Depth/room/baseline_0.25_3views"),
            # output_path=Path("./outputs/eval_images/w_Marigold_Depth/room/DS_Loss_0.25_3views_0"),
            # output_path=Path("./outputs/eval_images/w_Marigold_Depth/room/DS_Loss_0.25_3views_0_w_warping_loss"),
            # output_path=Path("./outputs/eval_images/w_Marigold_Depth/room/MSE_Loss_0.25_3views_0_w_warping_loss"),
            # output_path=Path("./outputs/eval_images/w_Marigold_Depth/room/DS_Loss_0.25_3views_2000"),
            # output_path=Path("./outputs/eval_images/wo_Marigold_Depth/3_half_data"),
            # output_path=Path("./outputs/eval_images/wo_Marigold_Depth/room/baseline_0.25"),
            # output_path=Path("./outputs/eval_images/w_Marigold_Depth/room/w_filter_outlier_0.25_3views"),
            # output_path=Path("./outputs/eval_images/w_Marigold_Depth/room/wo_filter_outlier_0.25"),
            # output_path=Path("./outputs/eval_images/wo_Marigold_Depth/DTU/DTU_scan24/baseline"),
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=NerfstudioDataParserConfig(
                    # scale_factor=0.25,    # used for llff dataset
                    depth_unit_scale_factor=1.0,
                    # train_split_fraction=0.07,
                    # train_split_fraction=0.9,
                    # flower
                    # train_split_fraction=0.27,
                    # train_split_fraction=0.87,
                    train_split_fraction=0.2,
                    # train_split_fraction=0.64,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NopeNerfactoModelConfig(
                predict_normals=True,
                # llff dataset
                # scale_init=10.0,
                # offset_init=5.0,
                # sheep dataset
                optim_scale_and_offset=True,
                # optim_scale_and_offset=False,
                scale_init=1.0,
                offset_init=0.0,
                depth_loss_mult=1e-3,
                depth_loss_type=NopeNerfDepthLossType.NOPE_NERF,
                # depth_loss_type=NopeNerfDepthLossType.RELATIVE_LOSS,
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
            "scale_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=200000),
            },
            "offset_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Nope-nerfacto training.",
)
