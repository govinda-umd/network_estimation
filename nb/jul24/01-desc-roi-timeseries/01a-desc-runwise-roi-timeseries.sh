#!/bin/bash

FILE_path=$1
FILE=$2
PARCELS=$3
TS_FOLDER=$4

main_dir="$(pwd)"

FILE=${FILE_path}/${FILE}
OUT_FILE=$(python -c "print('_'.join('${FILE}'.split('/')[-1].split('_')[:-1]))")_desc-roi-ts
echo "${OUT_FILE}"
cd "${TS_FOLDER}"

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
-prefix "${TS_FOLDER}/${OUT_FILE}.nii.gz"

3dROIstats -overwrite \
-quiet \
-mask "${PARCELS}" \
"${TS_FOLDER}/${OUT_FILE}.nii.gz" > "${TS_FOLDER}/${OUT_FILE}.txt"

rm -rf "${TS_FOLDER}/${OUT_FILE}.nii.gz"

cd "${main_dir}"