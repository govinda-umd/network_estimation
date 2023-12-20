#!/bin/bash

PARCELS=$1
file=$2
n=$3
k=$4

OUT_PATH=~/mouse_dataset/roi/svinet_folders/${PARCELS}/k"${k}"
mkdir -p "${OUT_PATH}"

svinet \
-file "${file}" \
-n "${n}" -k "${k}" \
-link-sampling

cd n"${n}"-k"${k}"-mmsb-linksampling
svinet \
-file "${file}" \
-n "${n}" -k "${k}" \
-gml

cd ../

out_file=$(python -c "print('_'.join('${file}'.split('/')[-1].split('_')[:-1]))")_k${k}
rm -r "${OUT_PATH}/${out_file}"
mv n"${n}"-k"${k}"-mmsb-linksampling "${OUT_PATH}/${out_file}"