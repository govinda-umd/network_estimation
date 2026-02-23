#!/bin/bash

# Define the base model-fits directory
BASE_MODEL_FITS="${HOME}/new_mouse_dataset/roi-results-v3/source-allen_space-ccfv2_braindiv-whl_nrois-172_res-200/graph-constructed/method-pearson/threshold-signed/edge-binary/density-20/layer-individual/unit-grp-boot/model-fits"

# Explicitly including the non-degree-corrected models you mentioned
CLEAN_MODELS=("sbm--m" "sbm--a" "sbm-dc-d" "sbm-dc-h" "sbm-dc-o" "sbm-nd-o" "sbm-nd-d" "sbm-nd-h")

for boot_dir in "$BASE_MODEL_FITS"/boot-*; do
    if [ -d "$boot_dir" ]; then
        echo "Processing $(basename "$boot_dir")..."
        for sbm in "${CLEAN_MODELS[@]}"; do
            target_path="$boot_dir/$sbm"
            if [ -d "$target_path" ]; then
                # Deleting everything except your diagnostics and evidence
                find "$target_path" -type f \
                   ! -name "desc-evidence.pkl" \
                   ! -name "desc-Bes-dls.pkl" \
                   -delete
            fi
        done
    fi
done
echo "Cleanup complete. Your HPC storage quota will thank you."