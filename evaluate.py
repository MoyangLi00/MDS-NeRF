import os
import numpy as np
import torch

from scipy.spatial.transform import Rotation

import evo.tools.file_interface as file_interface
import evo.core.trajectory as trajectory
import evo.core.metrics as metrics
import evo.tools.plot as plot
import evo.core.sync as sync
import matplotlib.pyplot as plt

## Extract camera poses and trajectory
# Change the path of results
gt_pose_path = 'data/scannet_scene0806/pose'
marigold_pose_path = 'outputs/scannet_scene0806/optimized/marigold/cam_pose'
frozen_pose_path = 'outputs/scannet_scene0806/optimized/leres/cam_pose'
opt_params_path = 'outputs/scannet_scene0806/optimized/marigold/optimized_params.pt'
tum_format_traj_dir = 'outputs/scannet_scene0806/evaluation' # Path to save the tum format trajectory file

opt_params = torch.load(opt_params_path)
rgb_ids = opt_params['optimized_image_ids']
n_frames = len(rgb_ids)

gt_trajectory = np.zeros([n_frames, 8])
est_trajectory = np.zeros([n_frames, 8])
frozen_trajectory = np.zeros([n_frames, 8])

for i, id in enumerate(rgb_ids):
    gt_pose = np.loadtxt(os.path.join(gt_pose_path, id + '.txt')) # matrix 4x4
    gt_R = gt_pose[:3, :3]
    gt_T = gt_pose[:3, 3]
    gt_quat = Rotation.from_matrix(gt_R).as_quat()
    gt_trajectory[i, 0] = id
    gt_trajectory[i, 1:4] = gt_T
    gt_trajectory[i, 4:] = gt_quat

    inv_est_pose = np.load(os.path.join(marigold_pose_path, id + '_opt.npy')) # matrix 4x4
    est_pose = np.linalg.inv(inv_est_pose)
    est_R = est_pose[:3, :3]
    est_T = est_pose[:3, 3]
    est_quat = Rotation.from_matrix(est_R).as_quat()
    est_trajectory[i, 0] = id
    est_trajectory[i, 1:4] = est_T
    est_trajectory[i, 4:] = est_quat

    inv_frozen_pose = np.load(os.path.join(frozen_pose_path, id + '_opt.npy')) # matrix 4x4
    frozen_pose = np.linalg.inv(inv_frozen_pose)
    frozen_R = frozen_pose[:3, :3]
    frozen_T = frozen_pose[:3, 3]
    frozen_quat = Rotation.from_matrix(frozen_R).as_quat()
    frozen_trajectory[i, 0] = id
    frozen_trajectory[i, 1:4] = frozen_T
    frozen_trajectory[i, 4:] = frozen_quat

# Save trajectory
os.makedirs(tum_format_traj_dir, exist_ok=True)
gt_traj_file = os.path.join(tum_format_traj_dir, 'gt_traj.txt')
frozen_traj_file = os.path.join(tum_format_traj_dir, 'frozenrecon_est_traj.txt')
marigold_est_traj_file = os.path.join(tum_format_traj_dir, 'marigold_est_traj.txt')
labels = '# id tx ty tz qx qy qz qw'

with open(gt_traj_file, 'w') as file:
    file.write(labels + '\n')
    for row in gt_trajectory:
        row_str = '{:d}'.format(int(row[0]))
        row_str += ' ' + ' '.join(['{:.3f}'.format(item) for item in row[1:]])
        file.write(row_str + '\n')
    file.close()
print(f"Data successfully written to '{gt_traj_file}'")

with open(marigold_est_traj_file, 'w') as file:
    file.write(labels + '\n')
    for row in est_trajectory:
        row_str = '{:d}'.format(int(row[0]))
        row_str += ' ' + ' '.join(['{:.3f}'.format(item) for item in row[1:]])
        file.write(row_str + '\n')
    file.close()
print(f"Data successfully written to '{marigold_est_traj_file}'")

with open(frozen_traj_file, 'w') as file:
    file.write(labels + '\n')
    for row in frozen_trajectory:
        row_str = '{:d}'.format(int(row[0]))
        row_str += ' ' + ' '.join(['{:.3f}'.format(item) for item in row[1:]])
        file.write(row_str + '\n')
    file.close()
print(f"Data successfully written to '{frozen_traj_file}'")


## Evaluate trajectory
gt_traj = file_interface.read_tum_trajectory_file(gt_traj_file)

# Marigold
marigold_est_traj = file_interface.read_tum_trajectory_file(marigold_est_traj_file)
marigold_est_traj, gt_traj = sync.associate_trajectories(marigold_est_traj, gt_traj)
aligned_marigold_est_traj = trajectory.align_trajectory(traj=marigold_est_traj, traj_ref=gt_traj, correct_scale=True)

marigold_ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
marigold_ape_metric.process_data((gt_traj, aligned_marigold_est_traj))
marigold_ape_stats = marigold_ape_metric.get_all_statistics()

marigold_rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_rad, delta=1, delta_unit=metrics.Unit.frames)
marigold_rpe_metric.process_data((gt_traj, aligned_marigold_est_traj))
marigold_rpe_stats = marigold_rpe_metric.get_all_statistics()

print("MarigoldPose Absolute Pose Error (Tran.) Statistics:")
print(marigold_ape_stats)

print("MarigoldPose Relative Pose Error (Rot.) Statistics:")
print(marigold_rpe_stats)

# FrozenRecon
frozen_est_traj = file_interface.read_tum_trajectory_file(frozen_traj_file)
frozen_est_traj, gt_traj = sync.associate_trajectories(frozen_est_traj, gt_traj)
aligned_frozen_est_traj = trajectory.align_trajectory(traj=frozen_est_traj, traj_ref=gt_traj, correct_scale=True)

frozen_ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
frozen_ape_metric.process_data((gt_traj, aligned_frozen_est_traj))
frozen_ape_stats = frozen_ape_metric.get_all_statistics()

frozen_rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_rad, delta=1, delta_unit=metrics.Unit.frames)
frozen_rpe_metric.process_data((gt_traj, aligned_frozen_est_traj))
frozen_rpe_stats = frozen_rpe_metric.get_all_statistics()

print("FrozenRecon Absolute Pose Error (Tran.) Statistics:")
print(frozen_ape_stats)

print("FrozenRecon Relative Pose Error (Rot.) Statistics:")
print(frozen_rpe_stats)

# Plot
fig = plt.figure(figsize=(9, 9))
ax = plot.prepare_axis(fig=fig, plot_mode=plot.PlotMode.xyz)
plot.traj(ax, plot.PlotMode.xyz, gt_traj, style='--', color='#009AD1', label='Ground Truth')#009AD1
plot.traj(ax, plot.PlotMode.xyz, aligned_frozen_est_traj, color='#9CCB86', label='FrozenRecon')
plot.traj(ax, plot.PlotMode.xyz, aligned_marigold_est_traj, color='#FC4E2A', label='Our Method')
title_str = f'3D Trajectory on ScanNet Scene0806'
ax.set_title(title_str, fontsize=24, weight='bold')
ax.set_zlim(1, 2)
plt.show()