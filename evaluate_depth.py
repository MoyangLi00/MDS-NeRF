import numpy as np
import os
import math
from matplotlib import pyplot as plt

dataset = "record3d_sheep"
eval_file_path = 'outputs/eval_images/Record3D/w_Marigold_Depth/' + dataset + '/27views_Marigold_depth/0010000'
Marigold_file_path = 'data/Record3D/'+ dataset + '/depths'
gt_file_path = 'data/Record3D/'+ dataset + '/depths_gt'
eval_res_save = 'outputs/eval_images/Record3D/eval_depth_results/' + dataset + '/27views'
os.makedirs(eval_res_save, exist_ok=True)

# train_split_fraction = 0.2
train_split_fraction = 0.64
# train_split_fraction = 0.27
# train_split_fraction = 0.87

depth_filenames = [f for f in os.listdir(gt_file_path) if f.endswith('.npy')]
depth_filenames.sort()
val_depth_filenames = [f for f in os.listdir(eval_file_path) if f.endswith('.npy') and "depth" in f]
val_depth_filenames.sort()

# split the dataset into training and evaluation sets
num_images = len(depth_filenames)
num_train_images = math.ceil(num_images * train_split_fraction)
num_eval_images = num_images - num_train_images
i_all = np.arange(num_images)
i_train = np.linspace(
    0, num_images - 1, num_train_images, dtype=int
)  # equally spaced training images starting and ending at 0 and num_images-1
i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
assert len(i_eval) == num_eval_images

scale = np.load(eval_file_path + '/' + 'scales.npy')
offset = np.load(eval_file_path + '/' + 'offsets.npy')


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = np.where((target / pred) > (pred / target), (target / pred), (pred / target))

    d1 = np.sum(thresh < 1.25).astype(np.float64) / len(thresh)
    d2 = np.sum(thresh < 1.25 ** 2).astype(np.float64) / len(thresh)
    d3 = np.sum(thresh < 1.25 ** 3).astype(np.float64) / len(thresh)

    diff = pred - target
    diff_log = np.log(pred) - np.log(target)

    abs_rel = np.mean(np.abs(diff) / target)
    sq_rel = np.mean(diff ** 2 / target)

    rmse = np.sqrt(np.mean(diff ** 2))
    
    rmse_log = np.sqrt(np.mean(diff_log ** 2))

    log10 = np.mean(np.abs(np.log10(pred) - np.log10(target)))
    silog = np.sqrt((diff_log ** 2).mean() - 0.5 * (diff_log.mean() ** 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10':log10.item(), 'silog':silog.item()}

# evaluate aligned_depth
align_img_idx = 0

train_RMSE_error = []
train_relative_error = []
for idx in i_train:
    gt_depth = np.load(gt_file_path + '/' + depth_filenames[idx])
    Marigold_depth = np.load(Marigold_file_path + '/' + depth_filenames[idx])
    Marigold_depth = Marigold_depth * scale[align_img_idx] + offset[align_img_idx]
    error_map = np.abs(gt_depth - Marigold_depth)
    # plot error map with bar
    # create new plt
    relative_error_map = np.abs(gt_depth - Marigold_depth) / gt_depth

    metrics = eval_depth(Marigold_depth, gt_depth)

    plt.figure()
    plt.imshow(error_map)
    plt.colorbar()
    plt.title('RMSE Error: %.4f m' % (metrics['rmse']))
    plt.savefig(eval_res_save + '/train_error_map_' + depth_filenames[idx].replace('.npy', '.png'), dpi=300)

    plt.figure()
    plt.imshow(relative_error_map)
    plt.colorbar()
    plt.title('Absolute Relative Error: %.4f' % (metrics['abs_rel']))
    plt.savefig(eval_res_save + '/train_relative_error_map_' + depth_filenames[idx].replace('.npy', '.png'), dpi=300)

    train_RMSE_error.append(metrics['rmse'])
    train_relative_error.append(metrics['abs_rel'])

    align_img_idx += 1

val_img_idx = 0

val_RMSE_error = []
val_relative_error = []
for idx in i_eval:
    gt_depth = np.load(gt_file_path + '/' + depth_filenames[idx])
    val_depth = np.load(eval_file_path + '/' + val_depth_filenames[val_img_idx])
    error_map = np.abs(gt_depth - val_depth)
    relative_error_map = np.abs(gt_depth - val_depth) / gt_depth

    metrics = eval_depth(val_depth, gt_depth)

    # plot error map with bar
    plt.figure()
    plt.imshow(error_map)
    plt.colorbar()
    plt.title('RMSE Error: %.4f m' % (metrics['rmse']))
    plt.savefig(eval_res_save + '/val_error_map_' + depth_filenames[idx].replace('.npy', '.png'), dpi=300)

    plt.figure()
    plt.imshow(relative_error_map)
    plt.colorbar()
    plt.title('Absolute Relative Error: %.4f' % (metrics['abs_rel']))
    plt.savefig(eval_res_save + '/val_relative_error_map_' + depth_filenames[idx].replace('.npy', '.png'), dpi=300)

    val_RMSE_error.append(metrics['rmse'])
    val_relative_error.append(metrics['abs_rel'])

    val_img_idx += 1

with open(eval_res_save + '/train_RMSE_error.txt', 'w') as f:
    for item in train_RMSE_error:
        f.write("%s\n" % item)
    f.write("Mean: %.4f\n" % np.mean(train_RMSE_error))

with open(eval_res_save + '/train_relative_error.txt', 'w') as f:
    for item in train_relative_error:
        f.write("%s\n" % item)
    f.write("Mean: %.4f\n" % np.mean(train_relative_error))

with open(eval_res_save + '/val_RMSE_error.txt', 'w') as f:
    for item in val_RMSE_error:
        f.write("%s\n" % item)
    f.write("Mean: %.4f\n" % np.mean(val_RMSE_error))

with open(eval_res_save + '/val_relative_error.txt', 'w') as f:
    for item in val_relative_error:
        f.write("%s\n" % item)
    f.write("Mean: %.4f\n" % np.mean(val_relative_error))

# evaluate rendered depth



