#!/bin/bash

FILE=$1
PARCELS=$2
TS_FOLDER=$3

base_dir="$(pwd)"

OUT_FILE=$(python -c "print('_'.join('${FILE}'.split('/')[-1].split('_')[:-1]))")_desc-roi-ts.txt
echo "${OUT_FILE}"

mask="$(cat ${FILE} | head -n2 | tail -n1)"
mask=${mask::-1}

data="$(cat ${FILE} | head -n3 | tail -n1)"
data=${data::-1}

cmask="~/mouse_dataset/voxel/common_brain_mask.nii.gz"

3dcalc -overwrite \
-a "${data}" \
-b "${mask}" \
-c "${cmask}" \
-expr "a * b * c" \
-prefix "${TS_FOLDER}"/ts.nii.gz

3dROIstats -overwrite \
-quiet \
-mask "${PARCELS}" \
"${TS_FOLDER}"/ts.nii.gz > "${TS_FOLDER}/${OUT_FILE}"

rm -rf "${TS_FOLDER}"/ts.nii.gz

cd "${base_dir}"