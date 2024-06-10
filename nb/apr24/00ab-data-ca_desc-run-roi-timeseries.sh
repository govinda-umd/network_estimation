#!/bin/bash

ALL_NII_FILES=$1
PARCELS=$2

# cat ${ALL_NII_FILES}

parallel --max-procs 17 \
bash 00aa-data-ca_desc-roi-timeseries.sh ::: `cat ${ALL_NII_FILES}` ::: ${PARCELS}