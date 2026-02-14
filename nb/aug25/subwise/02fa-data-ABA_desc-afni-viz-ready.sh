# 1. Make a temp folder for viewing
cd ~/lab-data
mkdir -p ~/lab-data/afni_viz_ready

# 2. Loop through your structure and create renamed links
# Adjust the list of subjects/conditions if needed
for sub in ABA602 ABA603 ABA604 ABA606 ABA607; do
    for cond in highT highR lowT lowR; do
        for level in 0 1; do
            
            # Define the source file path
            src="aba/NEWMAX_ROIs_final_gm_100_2mm/analysis-trial-end/graph-constructed/method-pearson/threshold-signed/edge-binary/density-20/layer-individual/unit-sub/cond-${cond}/estimates/individual/sub-${sub}/expected-marginal-visuals/nii/sbm-nd-h/mode-00_level-${level}.nii.gz"
            
            # Define a clean, descriptive name for AFNI
            # Example: s602_highT_L1.nii.gz
            dest=~/lab-data/afni_viz_ready/${sub}_${cond}_lev${level}.nii.gz
            
            # Create the link (only if file exists)
            if [ -f "$src" ]; then
                ln -s "$(pwd)/$src" "$dest"
            fi
            
        done
    done
done

# 3. Link your templates there too so everything is in one place
ln -s "$(pwd)/ROI_mask/NEWMAX_ROIs_final_gm_100_2mm.nii.gz" ~/lab-data/afni_viz_ready/00_Parcellation.nii.gz
ln -s "$(pwd)/ROI_mask/MNI152_T1_2mm_brain.nii.gz" ~/lab-data/afni_viz_ready/00_MNI_Template.nii.gz