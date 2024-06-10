#!/bin/bash

FILE=$1
PARCELS=$2

BAND=$(python -c "print('${FILE}'.split('/')[-2])")
ASR=$(python -c "print('_'.join('${FILE}'.split('/')[-1].split('_')[:3]))")

TS_FILE="${HOME}/new_mouse_dataset/roi_results/calcium/${BAND}/roi_timeseries/${ASR}"_desc-ts.txt
echo ${TS_FILE}

ls ${PARCELS}

3dROIstats -overwrite \
-quiet \
-mask "${PARCELS}" \
"${FILE}" > "${TS_FILE}"