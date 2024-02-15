#!/bin/bash

sub=$1
ses=$2

sub=$(printf "%02d" $sub)


REG_path=/home/govindas/mouse_dataset/voxel/regression_analysis

echo "========================================================="
echo "          Preprocessing Subject, Session: ${sub}; ${ses}"
echo "========================================================="

basis="CSPLIN(-10,15,26)"

# first stage regression

3dDeconvolve -overwrite \
-input "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-INPUT.nii.gz" \
-mask "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-MASK.nii.gz" \
-polort A \
-noFDR \
-local_times \
-censor "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-CENSOR.txt" \
-concat "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-CONCAT.1D" \
-num_stimts 1 \
-stim_times 1 "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-STIM.txt" ${basis}   -stim_label 1 LED \
-x1D "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-DESIGN_MAT.1D" \
-x1D_stop

# REMLfit
echo "****** 3dREMLfit ******"
3dREMLfit -matrix "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-DESIGN_MAT.1D" \
-input "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-INPUT.nii.gz" \
-noFDR \
-Rbuck "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-bucket_REML.nii.gz" \

echo "Well done, good bye!"