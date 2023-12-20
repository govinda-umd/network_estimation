#!/bin/bash

for sub in 1 2 3 4 5 6 7 8 9 10
do 
    for ses in 1 2 3
    do 
        bash unassumed_deconvolution_roi.sh ${sub} ${ses}
    done
done 