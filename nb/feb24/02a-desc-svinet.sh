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
mkdir -p "${OUT_PATH}"/"${SSR_FOLDER}"
# ssr_folder: (sub, ses, run)_folder

main_dir=$(pwd)
cd "${OUT_PATH}/${SSR_FOLDER}"

svinet \
-file "${FILE}" \
-n "${n}" -k "${k}" \
-link-sampling \
-seed "${seed}"

seed_folder=n"${n}"-k"${k}"-mmsb-seed"${seed}"-linksampling
echo "${OUT_PATH}/${SSR_FOLDER}/seed-$(leading_zero_fill 3 ${seed})"

mv "${OUT_PATH}/${SSR_FOLDER}/seed-$(leading_zero_fill 3 ${seed})" tmp
mv "${seed_folder}" "${OUT_PATH}/${SSR_FOLDER}/seed-$(leading_zero_fill 3 ${seed})"
rm -rf tmp

cd "${main_dir}"