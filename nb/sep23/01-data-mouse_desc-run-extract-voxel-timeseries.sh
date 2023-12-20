#!/bin/bash

FILE_path=~/mouse_dataset/voxel/all_file_collections

for file in $(ls ~/mouse_dataset/voxel/all_file_collections)
do 
    echo "${file}"
    echo "----------"
    bash 01-data-mouse_desc-extract-voxel-timeseries.sh "${FILE_path}/${file}"
    echo "----------"
done