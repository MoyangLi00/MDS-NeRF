#!/usr/bin/env bash
set -e
set -x

ckpt_dir=${ckpt_dir:-marigold/checkpoints}
mkdir -p $ckpt_dir
cd $ckpt_dir

checkpoint_name=marigold-v1-0

if [ -d $checkpoint_name ]; then
    exit 0
fi

wget -nv --show-progress https://share.phys.ethz.ch/~pf/bingkedata/marigold/checkpoint/${checkpoint_name}.tar
tar -xf ${checkpoint_name}.tar
rm ${checkpoint_name}.tar
