#!/bin/bash

REG_path=/home/govindas/mouse_dataset/voxel/regression_analysis

# for voxel clusters showing contrast
for sub in `seq 1 10`
do 
    for ses in `seq 1 3`
    do 
        sub=$(printf "%02d" $sub)
        3dTstat -overwrite \
        -mean \
        -prefix "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-ESTIMS_NOSTIM.nii.gz" \
        "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-bucket_REML.nii.gz"'[1..5]'

        3dTstat -overwrite \
        -mean \
        -prefix "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-ESTIMS_STIM.nii.gz" \
        "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-bucket_REML.nii.gz"'[11..15]'
    done
done 

3dttest++ -overwrite \
-paired \
-setA "${REG_path}/sub-SLC*_ses-*_desc-ESTIMS_STIM.nii.gz" \
-setB "${REG_path}/sub-SLC*_ses-*_desc-ESTIMS_NOSTIM.nii.gz" \
-labelA STIM \
-labelB NOSTIM \
-prefix "${REG_path}/desc-GROUP.nii.gz"


# for voxel responses during contrast
for sub in `seq 1 10`
do 
    for ses in `seq 1 3`
    do  
        sub=$(printf "%02d" $sub)

        3dbucket -overwrite \
        -prefix "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-RESP.nii.gz" \
        "${REG_path}/sub-SLC${sub}_ses-${ses}_desc-bucket_REML.nii.gz"'[1..26]'
    
    done
done

3dttest++ -overwrite \
-brickwise \
-setA "${REG_path}/sub-SLC*_ses-*_desc-RESP.nii.gz" \
-labelA RESP \
-prefix "${REG_path}/desc-GROUP-RESP.nii.gz"

3dbucket -overwrite \
-prefix "${REG_path}/desc-GROUP-RESP-MEAN.nii.gz" \
"${REG_path}/desc-GROUP-RESP.nii.gz"'[0..$(2)]'