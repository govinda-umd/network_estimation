#!/bin/bash

ALL_NII_FILES=$1
PARCELS=$2

# cat ${ALL_NII_FILES}

parallel --max-procs 17 \
bash 00ba-data-fmri_desc-roi-timeseries.sh ::: `cat ${ALL_NII_FILES}` ::: ${PARCELS}