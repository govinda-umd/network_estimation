#!/bin/bash

leading_zero_fill ()
{
    # print the number as a string with a given number of leading zeros
    printf "%0$1d\\n" "$2"
}

PARCELS=$1
ROIS_folder=$2
file=$3
n=$4
k=$5
seed=$6

OUT_PATH=~/mouse_dataset/roi/${ROIS_folder}/svinet_folders/${PARCELS}/k"${k}"
ssr_folder=$(python -c "print('_'.join('${file}'.split('/')[-1].split('_')[:-1]))")_k${k}
mkdir -p "${OUT_PATH}/${ssr_folder}"
# ssr_folder: (sub, ses, run)_folder

main_dir=$(pwd)
cd "${OUT_PATH}/${ssr_folder}"

svinet \
-file "${file}" \
-n "${n}" -k "${k}" \
-link-sampling \
-seed "${seed}"

seed_folder=n"${n}"-k"${k}"-mmsb-seed"${seed}"-linksampling
echo "${OUT_PATH}/${ssr_folder}/seed-$(leading_zero_fill 2 ${seed})"

mv "${OUT_PATH}/${ssr_folder}/seed-$(leading_zero_fill 2 ${seed})" tmp
mv "${seed_folder}" "${OUT_PATH}/${ssr_folder}/seed-$(leading_zero_fill 2 ${seed})"
rm -rf tmp

cd ${main_dir}