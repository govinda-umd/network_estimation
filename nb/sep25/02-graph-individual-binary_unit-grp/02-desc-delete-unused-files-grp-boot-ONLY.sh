#!/bin/bash

# 1. Read the target directory from the first argument
TARGET_BOOT="$1"

# Check if argument is provided
if [ -z "$TARGET_BOOT" ]; then
    echo "Error: No boot folder provided."
    echo "Usage: ./02-desc-delete-unused-files.sh /path/to/boot-000"
    exit 1
fi

# 2. Define the list of "remaining" SBM folders to clean
CLEAN_DIRS=("sbm-dc-d" "sbm-dc-h" "sbm-dc-o" "sbm-nd-o")

echo "Processing: $TARGET_BOOT"

for sbm in "${CLEAN_DIRS[@]}"; do
    target_path="${TARGET_BOOT%/}/$sbm"
    
    if [ -d "$target_path" ]; then
        # 3. Find files that are NOT 'evidence' AND NOT 'Bes-dls'
        # We chain two (! -name) flags. If a file matches EITHER name, it is excluded from deletion.
        
        # --- DRY RUN (Print only) ---
        # find "$target_path" -type f \
        #     ! -name "desc-evidence.pkl" \
        #     ! -name "desc-Bes-dls.pkl" \
        #     -print
        
        # --- ACTUAL DELETE (Uncomment below to run for real) ---
        find "$target_path" -type f \
           ! -name "desc-evidence.pkl" \
           ! -name "desc-Bes-dls.pkl" \
           -delete
        
    else
        : # Do nothing if folder doesn't exist
    fi
done