#!/bin/bash

FILE=$1
PARCELS=$2

ASR=$(python -c "asr = '${FILE}'.split('/')[-1].split('_'); print('_'.join([asr[0], asr[1], asr[4]]))")
echo ${ASR}

TS_FILE="${HOME}/new_mouse_dataset/roi_results/fmri/roi_timeseries/${ASR}_desc-ts.txt"
echo ${TS_FILE}

ls ${PARCELS}

3dROIstats -overwrite \
-quiet \
-mask "${PARCELS}" \
"${FILE}" > "${TS_FILE}"