import csv
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp 
import pickle 

from scipy import sparse, stats
import glob
from tqdm import tqdm
import ants
from nipype.interfaces import afni
import graph_tool.all as gt

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

# ==================

def collect_indiv_sbm_fits(args, files):
    def get(name):
        l = [s for s in ssr if name in s]
        return l[0].split('-')[-1] if len(l) > 0 else '0'
    
    fits_df = []
    for file in tqdm(files):
        ssr = file.split('/')[-2].split('_')
        sub, ses, run = list(map(get, ['sub', 'ses', 'run']))
        
        with open(f'{file}', 'rb') as f:
            all_vars = pickle.load(f)
            if len(all_vars) == 7:
                [g, L, pmode, modes, marginals, state, Bes] = all_vars
                converged = False
            elif len(all_vars) == 8:
                [g, L, pmode, modes, marginals, state, Bes, converged] = all_vars
        
        df = pd.DataFrame({
            'sub':[int(sub[-2:])],
            'ses':[int(ses)],
            'run':[int(run)],
            'ssr':[ssr],
            'graph':[g],
            'sbm':[f'sbm-{args.dc}-{args.sbm}'],
            'evidence':[L],
            'state':[state],
            'pmode':[pmode],
            'modes':[modes],
            'num_modes':[len(modes)],
            'marginals':[marginals],
            'Be':[Bes],
            'converged':[converged],
            'file':[file],
        })
        
        fits_df.append(df)
        
    fits_df = pd.concat(fits_df)
    fits_df = fits_df.sort_values(
        by=['sub', 'ses', 'run']
    ).reset_index(drop=True)

    return fits_df

# ==================

def collect_partitions(args, fits_df):
    bs = []
    for idx, row in tqdm(fits_df.iterrows()):
        sub, ses, run = row[['sub', 'ses', 'run']]
        modes = row['modes']
        M = len(row['pmode'].__getstate__()['bs'])

        for idx_mode, mode in enumerate(modes):
            bs += list(mode.get_partitions().values())

    pmode = gt.ModeClusterState(bs)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return bs, pmode

# ==================

def setup_args():
    class ARGS():
        pass

    args = ARGS()

    args.type = sys.argv[1] #'spatial'
    args.roi_size = int(sys.argv[2]) #225
    args.maintain_symmetry = sys.argv[3] == 'True'
    args.brain_div = sys.argv[4] #'whl'
    args.num_rois = int(sys.argv[5]) #162
    args.unit = sys.argv[6] # seswise
    args.denst = int(sys.argv[7]) # 10/15/20/25
    args.dc = sys.argv[8] == '1' # degree corrected?
    args.sbm = sys.argv[9] # p d o h
    # args.num_samples = int(sys.argv[10]) #5000
    args.SEED = int(sys.argv[10])
    
    return args

def main():
    args = setup_args()
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)

    DESC = (
        f'type-{args.type}'
        f'_size-{args.roi_size}'
        f'_symm-{args.maintain_symmetry}'
        f'_braindiv-{args.brain_div}'
        f'_nrois-{args.num_rois}'
    )
    
    BASE_path = f'{os.environ["HOME"]}/mouse_dataset'
    PARCELS_path = f'{BASE_path}/parcels'
    ROI_path = f'{BASE_path}/roi_results'
    ROI_RESULTS_path = f'{ROI_path}/{DESC}/{args.unit}/density-{args.denst}'
    FC_path = f'{ROI_RESULTS_path}/corr_mats'
    SBM_path = f'{ROI_RESULTS_path}/sbms'
    NPY_path = f'{ROI_RESULTS_path}/npy'
    os.system(f'mkdir -p {NPY_path}')
    NII_path = f'{ROI_RESULTS_path}/niis'
    os.system(f'mkdir -p {NII_path}/indiv')
    os.system(f'mkdir -p {NII_path}/group')

    if args.sbm == 'a' and args.dc == False:
        return None
    
    print(f'{SBM_path}')
    def modify_dc(args):
        dc = f'dc' if args.dc else f'nd'
        dc = f'' if args.sbm in ['a'] else dc
        args.dc = dc
        return args
    args = modify_dc(args)
    
    parcels_img = ants.image_read(f'{PARCELS_path}/{DESC}_desc-parcels.nii.gz')
    parcels = parcels_img.numpy()
    roi_labels = np.loadtxt(f'{PARCELS_path}/{DESC}_desc-labels.txt')
    
    files = sorted(glob.glob(f'{SBM_path}/*/sbm-{args.dc}-{args.sbm}*'))
    fits_df = collect_indiv_sbm_fits(args, files)
    
    # g = fits_df.iloc[9]['graph']
    # state = fits_df.iloc[9]['state']
    if args.sbm in ['h']:
        pass
        # dfs = collect_nested_partitions(args, fits_df)
    elif args.sbm in ['a', 'd']:
        bs, pmode = collect_partitions(args, fits_df)    
        
    with open(f'{NPY_path}/sbm-{args.dc}-{args.sbm}_desc-group-modes_partitions-all.npy', 'wb') as f:
        pickle.dump([bs, pmode], f)
    
    return None

# =========================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()