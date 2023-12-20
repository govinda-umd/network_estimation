#!/bin/bash

file=$1
PARCELS=$2
ROIS=$3
ROIS_folder=$4

OUT_PATH=~/mouse_dataset/roi/"${ROIS_folder}"/roi_timeseries_txt_files/"${PARCELS}"
mkdir -p "${OUT_PATH}"

mask="$(cat ${file} | head -n2 | tail -n1)"
mask=${mask::-1}

data="$(cat ${file} | head -n3 | tail -n1)"
data=${data::-1}

cmask="~/mouse_dataset/voxel/common_brain_mask.nii.gz"

3dcalc -overwrite \
-a "${data}" \
-b "${mask}" \
-c "${cmask}" \
-expr "a * b * c" \
-prefix ts.nii.gz

out_file=$(python -c "print('_'.join('${file}'.split('/')[-1].split('_')[:-1]))")_desc-roi-ts.txt

3dROIstats -overwrite \
-quiet \
-mask "${ROIS}" \
ts.nii.gz > "${OUT_PATH}/${out_file}"

rm -rf ts.nii.gz