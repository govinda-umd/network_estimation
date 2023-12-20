#!/bin/bash

file=$1

mask="$(cat ${file} | head -n2 | tail -n1)"
mask=${mask::-1}

data="$(cat ${file} | head -n3 | tail -n1)"
data=${data::-1}

ts="$(cat ${file} | head -n4 | tail -n1)"
ts=${ts::-1} # voxel ts file name

3dmaskdump -overwrite \
-mask "${mask}" \
-o "${ts}" \
"${data}"