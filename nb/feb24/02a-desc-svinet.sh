#!/bin/bash

leading_zero_fill ()
{
    # print the number as a string with a given number of leading zeros
    printf "%0$1d\\n" "$2"
}

FILE=$1
SVINET_FOLDER=$2

n=$3
k=$4
seed=$5

SSR_FOLDER=$(python -c "print('_'.join('${FILE}'.split('/')[-1].split('_')[:-1]))")
OUT_PATH="${SVINET_FOLDER}/k-${k}"
SSR_PATH="${OUT_PATH}"/"${SSR_FOLDER}"
mkdir -p "${SSR_PATH}"
# ssr_folder: (sub, ses, run)_folder

main_dir=$(pwd)
cd "${SSR_PATH}"

svinet \
-file "${FILE}" \
-n "${n}" -k "${k}" \
-link-sampling \
-seed "${seed}"

seed_folder=n"${n}"-k"${k}"-mmsb-seed"${seed}"-linksampling
SSRS_PATH="${SSR_PATH}/seed-$(leading_zero_fill 3 ${seed})"
echo "${SSRS_PATH}"

mv "${SSRS_PATH}" tmp
mv "${seed_folder}" "${SSRS_PATH}"
rm -rf tmp
# bcs, directly moving seed_folder to SSRS_PATH was not working

cd "${main_dir}"