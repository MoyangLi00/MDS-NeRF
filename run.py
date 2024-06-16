import os
import argparse
import glob
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
import logging

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

from utils.general_utils import extract_frames, ExcelEditor, sorted_files, pose_vec2mat, depth_project, get_ref_ids
from utils.lwlr_batch_utils import sparse_depth_lwlr_batch, fill_grid_sparse_depth_torch_batch

from marigold import MarigoldPipeline
from marigold.utils.image_util import colorize_depth_maps, chw2hwc


class MarigoldPose:
    def __init__(self, args) -> None:
        self.args = args

        # 1. Define parameters and file paths
        self.scene_name = args.scene_name
        self.model = self.args.depth_est_model

        self.output_path = os.path.join(args.output_path, self.scene_name)
        os.makedirs(self.output_path, exist_ok=True)

        # Difference between self.image_path and self.rgb_dir:
        # self.image_path is the source path; self.rgb_dir is the images used for MarigoldPose.
        # For images dataset, they are the same; for videos, frames extracted are saved in self.rgb_dir
        self.image_path = None
        if args.image_path is not None:
            self.image_path = os.path.join(args.image_path)
            self.rgb_dir = self.image_path

        if self.model == 'marigold':
            self.save_path = os.path.join(self.output_path, 'optimized/marigold/optimized_params.pt')
            self.workbook_save_path = os.path.join(self.output_path, 'optimized/marigold/optimize_time.xls')
        elif self.model == 'leres':
            self.save_path = os.path.join(self.output_path, 'optimized/leres/optimized_params.pt')
            self.workbook_save_path = os.path.join(self.output_path, 'optimized/leres/optimize_time.xls')

        self.outputs = {'times':{}}
        excel_title = ['scene_name', 'inference_time', 'optimize_time', 'total_time']
        self.excel = ExcelEditor('optimize_time', excel_title)

        # 2. Preprocess, extract RGB frames, set image_ids and rgb_paths
        self.preprocess()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            print('Warning: CUDA is not available. Running on CPU will be slow.')
        print(f'device = {self.device}')

        # 3. Estimate monocular depth using Marigold.
        print('estimating depth of each frame ...')
        self.depth_estimate()
        print('depth maps have been estimated.')

        # 4. Load data and initialize optimizer
        self.load_and_init()


    def preprocess(self):
        # 1. Record image_ids and rgb_paths
        self.rgb_ids = []
        self.rgb_paths = []
        for file in sorted_files(os.listdir(self.rgb_dir)):
            suffix = os.path.splitext(file)[-1]
            if suffix not in ['.jpg', '.png', '.JPG']:
                continue
            rgb_id = os.path.splitext(file)[0]
            self.rgb_ids.append(rgb_id)
            self.rgb_paths.append(os.path.join(self.rgb_dir, file))

        if len(self.rgb_ids) > 0:
            print(f'Found {len(self.rgb_ids)} images')
        else:
            print(f'Error: No image found in "{self.rgb_dir}"')
            exit(1)

        # 2. set resize parameter
        if self.args.resize_to_depth and args.image_path is not None:
            depth_path = os.path.join(os.path.dirname(os.path.normpath(self.rgb_dir)), 'depth')
            depth_file = sorted(os.listdir(depth_path), key=lambda x: int(os.path.basename(x)[:-4]))[0]
            depth_image = Image.open(os.path.join(depth_path, depth_file))
            self.w, self.h = depth_image.size
        else:
            self.w = self.args.width
            self.h = self.args.height

        # 3. resize and downsample frames according to frame interval or SSIM
        rgb_preprocess_dir = os.path.join(self.output_path, 'rgb')
        os.makedirs(rgb_preprocess_dir, exist_ok=True)
        rgb_preprocess_paths = []
        rgb_preprocess_ids = []

        interv = self.args.frame_sample_interv
        self.rgb_indexes = []

        print('down sampling images...')
        for i, rgb_path in enumerate(self.rgb_paths):
            if i % interv == 0:
                image = Image.open(rgb_path).convert("RGB")
                image = image.resize((self.w, self.h), resample=Image.Resampling.BILINEAR)
                rgb_preprocess_path = os.path.join(rgb_preprocess_dir, self.rgb_ids[i] + '.png')
                image.save(rgb_preprocess_path)
                rgb_preprocess_ids.append(self.rgb_ids[i])
                rgb_preprocess_paths.append(rgb_preprocess_path)
                self.rgb_indexes.append(i)
        self.rgb_ids = rgb_preprocess_ids
        self.rgb_paths = rgb_preprocess_paths
        self.n_images = len(self.rgb_ids)
        self.rgb_dir = rgb_preprocess_dir


    def depth_estimate(self):
        if self.model == 'marigold':
            self.output_path_depth_color = os.path.join(self.output_path, 'marigold/depth_colored')
            self.output_path_depth_bw = os.path.join(self.output_path, 'marigold/depth_bw')
            self.output_path_depth_npy = os.path.join(self.output_path, 'marigold/depth_npy')

            if os.path.exists(self.output_path_depth_npy):
                print('depth maps have been estimated, so will not estimate again.')
                self.outputs['times']['depth_estimate_time'] = 0
                return

            os.makedirs(self.output_path_depth_color, exist_ok=True)
            os.makedirs(self.output_path_depth_bw, exist_ok=True)
            os.makedirs(self.output_path_depth_npy, exist_ok=True)

            checkpoint_path = self.args.checkpoint

            denoise_steps = self.args.denoise_steps
            ensemble_size = self.args.ensemble_size
            if ensemble_size > 15:
                logging.warning('Running with large ensemble size will be slow.')
            half_precision = self.args.half_precision

            processing_res = self.args.processing_res
            match_input_res = not self.args.output_processing_res
            if 0 == processing_res and match_input_res is False:
                logging.warning(
                    'Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers.'
                )
            resample_method = self.args.resample_method

            color_map = self.args.color_map
            seed = self.args.seed
            batch_size = self.args.batch_size

            # -------------------- Marigold Model --------------------
            if half_precision:
                dtype = torch.float16
                variant = 'fp16'
                logging.info(
                    f'Running with half precision ({dtype}), might lead to suboptimal result.'
                )
            else:
                dtype = torch.float32
                variant = None

            marigold_pipe = MarigoldPipeline.from_pretrained(
                checkpoint_path, variant=variant, torch_dtype=dtype
            )

            try:
                marigold_pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                pass  # run without xformers
            marigold_pipe = marigold_pipe.to(self.device)

            # -------------------- Inference and saving --------------------
            with torch.no_grad():
                time_start = time.time()

                for rgb_path in tqdm(self.rgb_paths, desc='Estimating depth', leave=True):
                    # Read input image
                    input_image = Image.open(rgb_path)

                    # Random number generator
                    if seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(seed)

                    # Predict depth
                    pipe_out = marigold_pipe(
                        input_image,
                        denoising_steps=denoise_steps,
                        ensemble_size=ensemble_size,
                        processing_res=processing_res,
                        match_input_res=match_input_res,
                        batch_size=batch_size,
                        color_map=color_map,
                        show_progress_bar=True,
                        resample_method=resample_method,
                        generator=generator,
                    )

                    depth_pred: np.ndarray = pipe_out.depth_np
                    depth_colored: Image.Image = pipe_out.depth_colored

                    rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                    pred_name_base = rgb_name_base + '_pred'

                    # Save as npy
                    npy_save_path = os.path.join(self.output_path_depth_npy, f'{pred_name_base}.npy')
                    if os.path.exists(npy_save_path):
                        logging.warning(f'Existing file: "{npy_save_path}" will be overwritten')
                    np.save(npy_save_path, depth_pred)

                    # Save as 16-bit uint png
                    depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
                    png_save_path = os.path.join(self.output_path_depth_bw, f'{pred_name_base}.png')
                    if os.path.exists(png_save_path):
                        logging.warning(f'Existing file: "{png_save_path}" will be overwritten')
                    Image.fromarray(depth_to_save).save(png_save_path, mode='I;16')

                    # Colorize
                    colored_save_path = os.path.join(self.output_path_depth_color, f'{pred_name_base}_colored.png')
                    if os.path.exists(colored_save_path):
                        logging.warning(f'Existing file: "{colored_save_path}" will be overwritten')
                    depth_colored.save(colored_save_path)

                time_end = time.time()
                self.outputs['times']['depth_estimate_time'] = time_end - time_start

        elif self.model == 'leres':
            time_start = time.time()
            cmd = 'python ./leres/tools/test_depth.py --load_ckpt ./leres/res101.pth --backbone resnext101 --rgb_root %s --outputs %s' %(self.rgb_dir, os.path.join(self.output_path, 'leres'))
            os.system(cmd)
            self.output_path_depth_npy = os.path.join(self.output_path, 'leres/pred_depth_npy')
            time_end = time.time()
            self.outputs['times']['depth_estimate_time'] = time_end - time_start


    def load_and_init(self):
        # 1. Load RGB images
        self.rgb_imgs = torch.stack([
            pil_to_tensor( Image.open(rgb_path).convert("RGB") ) for rgb_path in self.rgb_paths
        ], dim=0).to(device=self.device) # [n, 3, h, w]

        # 2. Read estimated depth maps
        if self.model == 'marigold':
            self.mono_depths = torch.stack([
                torch.from_numpy(
                    np.load(os.path.join(self.output_path_depth_npy, rgb_id + '_pred.npy')).astype(np.float32),
                    )[None, ...] for rgb_id in self.rgb_ids
                ], dim=0).float().to(device=self.device)

        elif self.model == 'leres':
            self.mono_depths = torch.stack([
                torch.from_numpy(
                    np.load(os.path.join(self.output_path_depth_npy, rgb_id + '-depth.npy')).astype(np.float32),
                    )[None, ...] for rgb_id in self.rgb_ids
                ], dim=0).float().to(device=self.device)

        # 3. Set optimizer
        self.optimize_params = {}
        params = []

        pose_lr_t = self.args.pose_lr_t[0]
        pose_lr_r = self.args.pose_lr_r[0]
        focal_lr = self.args.focal_lr[0]
        depth_lr = self.args.depth_lr[0]
        sparse_points_lr = self.args.sparse_points_lr[0]

        # cam pose settings
        poses_6dof_t = torch.zeros((self.n_images-1, 3)).double().cuda()
        poses_6dof_r = torch.zeros((self.n_images-1, 3)).double().cuda()

        self.optimize_params['poses_6dof_t'] = poses_6dof_t
        self.optimize_params['poses_6dof_r'] = poses_6dof_r
        params.append({'params': self.optimize_params['poses_6dof_t'], 'lr': pose_lr_t})
        params.append({'params': self.optimize_params['poses_6dof_r'], 'lr': pose_lr_r})

        # focal length settings
        self.optimize_params['focal_length_ratio'] = torch.tensor([1.2]).cuda().double()
        params.append({'params': [self.optimize_params['focal_length_ratio']], 'lr': focal_lr})

        # depth scale and shift settings
        self.optimize_params['scale_map'] = torch.tensor([1.]).cuda().repeat(self.n_images)
        self.optimize_params['shift_map'] = torch.tensor([0.]).cuda().repeat(self.n_images)

        params.append({'params': [self.optimize_params['scale_map'], self.optimize_params['shift_map']], 'lr': depth_lr})

        self.optimize_params['sparse_guided_points'] = torch.ones((self.n_images, 5, 5)).float().cuda()
        params.append({'params': [self.optimize_params['sparse_guided_points']], 'lr': sparse_points_lr})

        for key in self.optimize_params:
            self.optimize_params[key].requires_grad = True

        if params != []:
            self.optimizer = torch.optim.AdamW(params,
                                        lr=0,
                                        betas=(0.9, 0.999),
                                        weight_decay=0)
        else:
            raise ValueError('Error of training: No params to train.')


    def optimize(self):
        time_start = time.time()

        for epoch in range(self.args.epochs):
            # 1. Set learning rate
            pose_lr_t = self.args.pose_lr_t[epoch]
            pose_lr_r = self.args.pose_lr_r[epoch]
            focal_lr = self.args.focal_lr[epoch]
            depth_lr = self.args.depth_lr[epoch]
            sparse_points_lr = self.args.sparse_points_lr[epoch]

            lr_set_cnt = 0
            # cam pose
            self.optimizer.param_groups[lr_set_cnt]['lr'] = pose_lr_t
            lr_set_cnt += 1
            self.optimizer.param_groups[lr_set_cnt]['lr'] = pose_lr_r
            lr_set_cnt += 1
            print('Poses lr set...')

            # focal length
            self.optimizer.param_groups[lr_set_cnt]['lr'] = focal_lr
            lr_set_cnt += 1
            print('Intrinsic lr set...')

            # depth scale and shift
            self.optimizer.param_groups[lr_set_cnt]['lr'] = depth_lr
            lr_set_cnt += 1
            self.optimizer.param_groups[lr_set_cnt]['lr'] = sparse_points_lr
            lr_set_cnt += 1
            print('Depth lr set...')

            # 2. Optimization
            progress_bar = tqdm(range(0, self.args.iters_per_epoch), desc="Training progress of epoch %s" %epoch)
            for t in range(self.args.iters_per_epoch):

                # 2.1 compute camera intrinsic
                fx = self.optimize_params['focal_length_ratio'] * ((self.h + self.w) / 2)
                fy = self.optimize_params['focal_length_ratio'] * ((self.h + self.w) / 2)
                intrinsics = torch.tensor([[[0, 0, self.w/2], [0, 0, self.h/2], [0, 0, 1]]]).cuda().double()
                intrinsics[0, 0:1, 0:1] = fx
                intrinsics[0, 1:2, 1:2] = fy

                # 2.2 compute global camera poses
                poses_6dof = torch.cat((self.optimize_params['poses_6dof_t'], self.optimize_params['poses_6dof_r']), dim=1)
                pose_init = torch.eye(4)[None, ...].double().cuda()
                pose = torch.eye(4).double().cuda()
                poses_computed_global = [pose_init]
                for i in range(self.n_images-1):
                    pose_temp = pose_vec2mat(poses_6dof[i][None, ...])
                    pose_ones = torch.tensor([[[0, 0, 0, 1]]]).double().cuda()
                    pose_temp = torch.cat((pose_temp, pose_ones), dim=1)
                    pose_init = pose_temp @ pose_init
                    poses_computed_global.append(pose_init)
                poses_computed_global = torch.cat(poses_computed_global, dim=0)

                # 2.3 compute relative camera poses of each two frames
                if epoch == 0:
                    rel_angle = None
                else:
                    poses_computed_global_i = poses_computed_global[:, None] # [n, 1, 4, 4]
                    poses_computed_global_j = poses_computed_global[None, ...]  # [1, n, 4, 4]
                    poses_computed_relative = poses_computed_global_j.double() @ poses_computed_global_i.double().inverse()

                    R_rel_trace = poses_computed_relative[:, :, 0, 0] + poses_computed_relative[:, :, 1, 1] + poses_computed_relative[:, :, 2, 2] # R_rel_trace=2cosw+1
                    eps = 1e-7
                    rel_angle = torch.acos(torch.clamp((R_rel_trace - 1) / 2, -1 + eps, 1 - eps)) # relative angle
                    rel_angle[(rel_angle - rel_angle.T) > eps] = 3.14

                # 2.4 sample tgt_ids and ref_ids for warping and computing losses
                tgt_ids = []
                ref_ids = []
                sample_imgs = self.args.samples_per_iter
                sample_imgs = min(sample_imgs, self.n_images)
                angle = None if epoch == 0 else rel_angle.clone()
                sample_ref_num = 1

                tgt_ids = torch.randperm(self.n_images)[:sample_imgs].cuda()
                ref_ids = get_ref_ids(tgt_ids, self.n_images, self.args.near_frames_num, angle=angle, epoch=epoch, sample_ref_num=sample_ref_num, angle_thr=self.args.angle_thr)
                tgt_ids = tgt_ids[:, None].repeat(1, sample_ref_num).view(-1)
                ref_ids = ref_ids.view(-1)
                assert tgt_ids.shape == ref_ids.shape

                sample_num = int(self.w * self.h * self.args.sample_pix_ratio)
                valid_mask = (ref_ids >= 0) & (ref_ids <= self.n_images-1) & \
                    ((torch.isnan(poses_computed_global[tgt_ids.clamp(min=0, max=self.n_images-1)]) + torch.isnan(poses_computed_global[ref_ids.clamp(min=0, max=self.n_images-1)])).sum(dim=(1, 2)) == 0)
                tgt_ids = tgt_ids[valid_mask]
                ref_ids = ref_ids[valid_mask]
                n = ref_ids.shape[0]

                # 2.5 sample pixels from coords
                coords_x = torch.arange(0, self.w).float().cuda()[None, None, :].repeat(n, self.h, 1).view(n, -1)
                coords_y = torch.arange(0, self.h).float().cuda()[None, :, None].repeat(n, 1, self.w).view(n, -1)

                valid_imgs_num = coords_x.shape[0]
                coords_mask_index = torch.randperm(self.h * self.w)[:sample_num].repeat(valid_imgs_num, 1).cuda()
                coords_mask = (coords_x != coords_x)
                coords_mask.scatter_(1, coords_mask_index, 1)

                coords_x = coords_x[coords_mask].view(valid_imgs_num, -1)
                coords_y = coords_y[coords_mask].view(valid_imgs_num, -1)

                # 2.6 computing local alignment linear regression of mono depth
                metric_depths_global = self.mono_depths * self.optimize_params['scale_map'][:, None, None, None].abs() + self.optimize_params['shift_map'][:, None, None, None]
                metric_depths_global = metric_depths_global.squeeze()
                metric_depths_global += 1e-6

                sparse_guided_depth = fill_grid_sparse_depth_torch_batch(metric_depths_global, self.optimize_params['sparse_guided_points'].abs(), fill_coords=None, device=torch.device('cuda'))
                sparse_guided_depth = sparse_guided_depth * metric_depths_global
                k_para = self.args.k_para
                metric_depths = sparse_depth_lwlr_batch(metric_depths_global, sparse_guided_depth, down_sample_scale=32, k_para=k_para, sample_num=sparse_guided_depth[0][sparse_guided_depth[0]>0].numel(), device=torch.device('cuda'))[:, None, ...]

                metric_depths[metric_depths < 0] = 0

                # 2.7 computing relative poses
                if epoch == 0:
                    poses = poses_computed_global[ref_ids] @ poses_computed_global[tgt_ids].inverse()
                else:
                    poses = poses_computed_relative[tgt_ids, ref_ids]

                assert (poses.sum(dim=1).sum(dim=1) == 0).sum() == 0

                # 2.8 warp from reference images to target images and compute losses
                tgt_sample_coords = torch.stack([coords_x, coords_y], dim=2)[:, :, None, :] # [n(filtered), sampled_num, 1, 2]
                tgt_sample_coords[..., 0] = 2 * tgt_sample_coords[..., 0] / (self.w - 1) - 1
                tgt_sample_coords[..., 1] = 2 * tgt_sample_coords[..., 1] / (self.h - 1) - 1

                intrinsics_tgt = intrinsics

                depth_computed_points, projected_x, projected_y = depth_project(coords_x, coords_y, metric_depths[tgt_ids].view(tgt_ids.numel(), 1, -1), intrinsics_tgt, poses, coords_mask, (self.h, self.w))
                ## valid_mask of projection
                valid_mask = (projected_x >= -0.95) & (projected_x <= 0.95) & (projected_y >= -0.95) & (projected_y <= 0.95)
                ## tgt color
                color_tgt = F.grid_sample(self.rgb_imgs[tgt_ids].float(), tgt_sample_coords.float(), padding_mode='zeros', align_corners=False)
                ## projected ref color
                sample_coords = torch.stack([projected_x, projected_y], dim=2)[:, :, None, :] # [n(filtered), sample_num, 1, 2]
                color_projected = F.grid_sample(self.rgb_imgs[ref_ids].float(), sample_coords.float(), padding_mode='zeros', align_corners=False)
                ## projected ref depth
                depth_projected_points = F.grid_sample(metric_depths[ref_ids].float(), sample_coords.float(), padding_mode='zeros', align_corners=False)
                depth_projected_points = depth_projected_points[:, 0, :, 0] # [n, sample_num]

                ## update valid_mask
                valid_mask[depth_projected_points == 0] = 0
                valid_mask[depth_computed_points == 0] = 0
                valid_mask[(color_projected.sum(dim=1) == 0)[..., 0]] = 0

                ## filter out valid depth
                depth_projected_points = depth_projected_points[valid_mask]
                depth_computed_points = depth_computed_points[valid_mask]
                assert depth_projected_points.shape == depth_computed_points.shape
                ## filter out valid colors
                color_tgt[~valid_mask[:, None, :, None].repeat(1, 3, 1, 1)] = 0
                color_projected[~valid_mask[:, None, :, None].repeat(1, 3, 1, 1)] = 0
                color_tgt = color_tgt.permute(0, 2, 1, 3)
                color_projected = color_projected.permute(0, 2, 1, 3)
                color_tgt = color_tgt[valid_mask]
                color_projected = color_projected[valid_mask]
                ## compute losses
                loss_photometric = (color_tgt - color_projected).abs()
                loss_photometric = loss_photometric.mean(dim=1).squeeze()
                loss_geometric = ((depth_projected_points - depth_computed_points).abs() / (depth_projected_points + depth_computed_points))

                valid_percent = self.args.max_valid_percent_loss
                if epoch < 2 and valid_percent < 1:
                    photo_thr = torch.quantile(loss_photometric, valid_percent)
                    loss_photometric = loss_photometric[loss_photometric < photo_thr]

                    geo_thr = torch.quantile(loss_geometric, valid_percent)
                    loss_geometric = loss_geometric[loss_geometric < geo_thr]

                loss_photometric = loss_photometric.mean()
                loss_geometric = loss_geometric.mean()
                loss_scale_norm = ((self.optimize_params['sparse_guided_points'][tgt_ids] - 1).abs().sum() + (self.optimize_params['sparse_guided_points'][ref_ids] - 1).abs().sum()) / (tgt_ids.numel() + ref_ids.numel())

                if t % self.args.print_iters == 0:
                    print_info = {}
                    print_info["Epoch"] = epoch
                    print_info["loss_photometric"] = loss_photometric.tolist()
                    print_info["loss_geometric"] = loss_geometric.tolist()
                    print_info["loss_scale_norm"] = loss_scale_norm.tolist()
                    progress_bar.set_postfix(print_info)
                    progress_bar.update(self.args.print_iters)
                if t == (self.args.iters_per_epoch - 1):
                    progress_bar.close()

                pc_weight = self.args.loss_pc_weight
                gc_weight = self.args.loss_gc_weight
                scale_norm_weight = self.args.loss_norm_weight

                loss = pc_weight[epoch] * loss_photometric + gc_weight[epoch] * loss_geometric
                loss += (scale_norm_weight[epoch] * loss_scale_norm)

                # 2.9 save for the latest iteration
                if t == (self.args.iters_per_epoch - 1) and (epoch == self.args.epochs - 1):
                    if self.model == 'marigold':
                        output_path_depth_npy_opt = os.path.join(self.output_path, 'optimized/marigold/depth/npy')
                        output_path_depth_color_opt = os.path.join(self.output_path, 'optimized/marigold/depth/color')
                        output_path_pose_opt = os.path.join(self.output_path, 'optimized/marigold/cam_pose')
                        output_path_intrinsic_opt = os.path.join(self.output_path, 'optimized/marigold/cam_intrinsic')
                    elif self.model == 'leres':
                        output_path_depth_npy_opt = os.path.join(self.output_path, 'optimized/leres/depth/npy')
                        output_path_depth_color_opt = os.path.join(self.output_path, 'optimized/leres/depth/color')
                        output_path_pose_opt = os.path.join(self.output_path, 'optimized/leres/cam_pose')
                        output_path_intrinsic_opt = os.path.join(self.output_path, 'optimized/leres/cam_intrinsic')
                    os.makedirs(output_path_depth_npy_opt, exist_ok=True)
                    os.makedirs(output_path_depth_color_opt, exist_ok=True)
                    os.makedirs(output_path_pose_opt, exist_ok=True)
                    os.makedirs(output_path_intrinsic_opt, exist_ok=True)

                    resized_height_width = np.array([self.h, self.w])

                    if intrinsics.shape[0] == 1:
                        intrinsics = intrinsics.repeat(self.rgb_imgs.shape[0], 1, 1)

                    # aligned depth
                    metric_depths_global = self.mono_depths * self.optimize_params['scale_map'][:, None, None, None].abs() + self.optimize_params['shift_map'][:, None, None, None]
                    metric_depths_global = metric_depths_global.squeeze()
                    metric_depths_global += 1e-8

                    # local alignment linear regression
                    sparse_guided_depth = fill_grid_sparse_depth_torch_batch(metric_depths_global, self.optimize_params['sparse_guided_points'].abs(), fill_coords=None, device=self.device)
                    sparse_guided_depth = sparse_guided_depth * metric_depths_global

                    k_para = self.args.k_para
                    optimized_depth = sparse_depth_lwlr_batch(metric_depths_global, sparse_guided_depth, down_sample_scale=32, k_para=k_para, sample_num=sparse_guided_depth[0][sparse_guided_depth[0]>0].numel(), device=torch.device('cuda')).squeeze() # lwlr
                    optimized_depth[optimized_depth < 0] = 0

                    optimized_params = dict(
                        optimized_depth = optimized_depth.detach().cpu().numpy(),
                        poses_computed_global = poses_computed_global.detach().cpu().numpy(),
                        optimized_intrinsic = intrinsics.detach().cpu().numpy(),
                        optimized_image_indexes = self.rgb_indexes,
                        optimized_image_ids = self.rgb_ids,
                        resized_height_width = resized_height_width,
                        optimized_scale_map = self.optimize_params['scale_map'].detach().cpu().numpy(),
                        optimized_shift_map = self.optimize_params['shift_map'].detach().cpu().numpy(),
                        optimized_sparse_guided_points = self.optimize_params['sparse_guided_points'].detach().cpu().numpy(),
                    )

                    # find reasonable near and far plane for visualization
                    nearest_depth = np.min(optimized_params['optimized_depth'][...])
                    farest_depth = np.max(optimized_params['optimized_depth'][...])
                    near_plane = nearest_depth - 0.2 * (farest_depth - nearest_depth)
                    far_plane = farest_depth + 0.2 * (farest_depth - nearest_depth)

                    for i, id in enumerate(self.rgb_ids):
                        depth_npy = optimized_params['optimized_depth'][i, ...]
                        depth_color = colorize_depth_maps(depth_npy, near_plane, far_plane, cmap=self.args.color_map).squeeze()  # [3, H, W], value in (0, 1)
                        depth_color = (depth_color * 255).astype(np.uint8)
                        depth_color = Image.fromarray(chw2hwc(depth_color))

                        cam_pose = optimized_params['poses_computed_global'][i, ...]
                        cam_intrinsic = optimized_params['optimized_intrinsic'][i, ...]

                        depth_npy_save_path = os.path.join(output_path_depth_npy_opt, f'{id}_opt.npy')
                        depth_color_save_path = os.path.join(output_path_depth_color_opt, f'{id}_opt.png')
                        cam_pose_save_path = os.path.join(output_path_pose_opt, f'{id}_opt.npy')
                        cam_intrinsic_save_path = os.path.join(output_path_intrinsic_opt, f'{id}_opt.npy')

                        np.save(depth_npy_save_path, depth_npy)
                        depth_color.save(depth_color_save_path)
                        np.save(cam_pose_save_path, cam_pose)
                        np.save(cam_intrinsic_save_path, cam_intrinsic)

                    self.outputs['optimized_params'] = optimized_params
                    torch.save(optimized_params, self.save_path)
                    if self.model == 'marigold':
                        print('Saved optimized depth results to:', os.path.join(self.output_path, 'optimized/marigold'))
                    elif self.model == 'leres':
                        print('Saved optimized depth results to:', os.path.join(self.output_path, 'optimized/leres'))
                    print('Saved optimized parameters to:', self.save_path)
                    break

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        time_end = time.time()
        self.outputs['times']['optimize_time'] = time_end - time_start
        self.outputs['times']['total_time'] = self.outputs['times']['depth_estimate_time'] + self.outputs['times']['optimize_time']


    def save_excel(self):
        column_data = [
                self.scene_name,
                self.outputs['times']['depth_estimate_time'],
                self.outputs['times']['optimize_time'],
                self.outputs['times']['total_time']
        ]
        self.excel.add_data(column_data)
        self.excel.save_excel(self.workbook_save_path)


    def run(self):
        self.optimize()
        self.save_excel()
        print('finished optimization.')

        return self.outputs



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_est_model', help='The type of model used for depth estimation (marigold or leres).', type=str)
    parser.add_argument('--image_path', help='The path to images for optimization.', type=str)
    parser.add_argument('--output_path', help='Output folders', type=str, default='./outputs')

    parser.add_argument('--resize_to_depth',  help='Whether resize rgb image to gt depth map.', action='store_true')
    parser.add_argument('--width', help='The width after resizing.', type=int, default=640)
    parser.add_argument('--height', help='The height after resizing.', type=int, default=480)

    parser.add_argument('--recon_voxel_size', help='Voxel size of reconstruction.', type=float, default=0.1)
    parser.add_argument('--angle_thr', help='Angle threshold of keyframe selection strategy.', type=float, default=np.pi/4)
    parser.add_argument('--iters_per_epoch', help='Iterations for each epoch. 2000 is usually more than enough, and should not be too small.', type=int, default=2000)

    parser.add_argument('--pose_lr_t', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-2, 1e-3, 1e-3])
    parser.add_argument('--pose_lr_r', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-3, 1e-4, 1e-4])
    parser.add_argument('--focal_lr', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-2, 1e-2, 1e-2])
    parser.add_argument('--depth_lr', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-1, 1e-2, 1e-2])
    parser.add_argument('--sparse_points_lr', help="hyperparameter of lr.", type=float, nargs='+', default=[1e-1, 1e-2, 1e-2])

    parser.add_argument('--loss_pc_weight', help='hyperparameter of losses.', type=float, nargs='+', default=[2, 2, 2])
    parser.add_argument('--loss_gc_weight', help='hyperparameter of losses.', type=float, nargs='+', default=[0.5, 1, 0.1])
    parser.add_argument('--loss_norm_weight', help='hyperparameter of losses.', type=float, nargs='+', default=[0.01, 0.1, 0.1])

    parser.add_argument('--frame_sample_interv', help='The downsample interval of frames.', type=int, default=1)
    parser.add_argument('--epochs', help='Total optimization epochs', type=int, default=3)
    parser.add_argument('--near_frames_num', help='Near frames for the first local optimization stage.', type=int, default=3)
    parser.add_argument('--samples_per_iter', help='Number of sample images for computing losses per iteration.', type=int, default=50)
    parser.add_argument('--sample_pix_ratio', help='sample partial pixels for reducing computation cost.', type=float, default=0.25) # the ratio does not matter so much
    parser.add_argument('--k_para', help='The k parameter of lwlr function.', type=int, default=50) # seems does not matter so much
    parser.add_argument('--max_valid_percent_loss', help='Filter out big loss values larger than this percentage. (0.9 means 90%)', type=float, default=0.9) # slightly improved
    parser.add_argument('--print_iters', help='The frequence of printing.', type=int, default=1)

    # Followings are Marigold parameters:
    parser.add_argument('--checkpoint', type=str, default='marigold/checkpoints/marigold-v1-0', help='Marigold checkpoint path')

    # inference setting
    parser.add_argument(
        '--denoise_steps',
        type=int,
        default=10,
        help='Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, recommended to use 10-50 steps, while for LCM 1-4 steps.',
    )
    parser.add_argument(
        '--ensemble_size',
        type=int,
        default=10,
        help='Number of predictions to be ensembled, more inference gives better results but runs slower.',
    )
    parser.add_argument(
        '--half_precision',
        '--fp16',
        action='store_true',
        help='Run with half-precision (16-bit float), might lead to suboptimal result.',
    )

    # resolution setting
    parser.add_argument(
        '--processing_res',
        type=int,
        default=768,
        help='Maximum resolution of processing. 0 for using input image resolution. Default: 768.',
    )
    parser.add_argument(
        '--output_processing_res',
        action='store_true',
        help='When input is resized, out put depth at resized operating resolution. Default: False.',
    )
    parser.add_argument(
        '--resample_method',
        choices=['bilinear', 'bicubic', 'nearest'],
        default='bilinear',
        help='Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`',
    )

    # depth map colormap
    parser.add_argument(
        '--color_map',
        type=str,
        default='Spectral',
        help='Colormap used to render depth predictions.',
    )

    # other settings
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Reproducibility seed. Set to `None` for unseeded inference.',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=0,
        help='Inference batch size. Default: 0 (will be set automatically).',
    )

    args = parser.parse_args()

    # images
    if args.image_path is not None:
        if args.image_path.split('/')[-1] == '':
            scene_name = args.image_path.split('/')[-3]
        else:
            scene_name = args.image_path.split('/')[-2]
    else:
        raise ValueError('Unsupported data type.')

    args.scene_name = scene_name

    print(f'scene name: "{args.scene_name}"')
    print(f'image path: "{args.image_path}"')

    if args.depth_est_model == 'marigold':
        print('Use Marigold')
        print('Marigold checkpoint:', args.checkpoint)
        save_params_path = os.path.join(args.output_path, args.scene_name, 'optimized/marigold/optimized_params.pt')
    elif args.depth_est_model == 'leres':
        print('Use LeReS')
        save_params_path = os.path.join(args.output_path, args.scene_name, 'optimized/leres/optimized_params.pt')
    else:
        raise ValueError('Unsupported model type.')

    if not os.path.exists(save_params_path):
        recon_optimizer = MarigoldPose(args)
        outputs = recon_optimizer.run()
        optimized_params = outputs['optimized_params']
        del recon_optimizer
