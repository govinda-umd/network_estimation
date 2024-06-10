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


# =================

def collect_sbm_fits(args, files):
    def get(name):
        l = [s for s in ssr if name in s]
        return l[0].split('-')[-1] if len(l) > 0 else '0'
    
    fits_df = []
    for file in tqdm(files):
        ssr = file.split('/')[-2].split('_')
        sub, ses, run = list(map(get, ['sub', 'ses', 'run']))
        
        with open(f'{file}', 'rb') as f:
            all_vars = pickle.load(f)
            if len(all_vars) == 10:
                [g, L, pmode, modes, marginals, state, bs, Bs, Bes, dls] = all_vars
                converged = False
            elif len(all_vars) == 11:
                [g, L, pmode, modes, marginals, state, bs, Bs, Bes, dls, converged] = all_vars
        
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
            'marginals':[marginals],
            'bs':[bs],
            'Bs':[Bs],
            'Be':[Bes],
            'dls':[dls],
            'converged':[converged],
            'file':[file],
        })
        
        fits_df.append(df)
        
    fits_df = pd.concat(fits_df)
    fits_df = fits_df.sort_values(
        by=['sub', 'ses', 'run']
    ).reset_index(drop=True)

    return fits_df

# =================

def collect_nested_partitions(args, fits_df):
    dfs = []
    for idx, row in tqdm(fits_df.iterrows()):
        sub, ses, run = row[['sub', 'ses', 'run']]
        modes = row['modes']
        M = len(row['bs'])

        for idx_mode, mode in enumerate(modes):
            # plausibility
            pls = int(args.num_samples*(mode.get_M() / M))
            for s in range(pls):
                # sample <pls> partiitions from mode
                b = mode.sample_nested_partition()

                df = pd.DataFrame({
                    'sub':[sub],
                    'ses':[ses],
                    'run':[run],
                    'mode':[idx_mode],
                    'sample':[s],
                    'b':[b],
                })
                dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)

    pmode = gt.ModeClusterState(dfs['b'].to_list(), nested=True)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return dfs, pmode
    
def collect_partitions(args, fits_df):
    dfs = []
    for idx, row in tqdm(fits_df.iterrows()):
        sub, ses, run = row[['sub', 'ses', 'run']]
        modes = row['modes']
        M = len(row['bs'])

        for idx_mode, mode in enumerate(modes):
            # plausibility
            pls = int(args.num_samples*(mode.get_M() / M))
            for s in range(pls):
                # sample <pls> partiitions from mode
                b = mode.sample_partition()

                df = pd.DataFrame({
                    'sub':[sub],
                    'ses':[ses],
                    'run':[run],
                    'mode':[idx_mode],
                    'sample':[s],
                    'b':[b],
                })
                dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)

    pmode = gt.ModeClusterState(dfs['b'].to_list())
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return dfs, pmode

def he_to_v(args, g, state, he_b):
    # half-edge partition to vertex partition
    v_b = np.zeros((g.num_vertices()), dtype=np.int32)
    for v, hes in zip(g.iter_vertices(), state.half_edges):
        try:
            v_b[v] = stats.mode(he_b[hes])[0][0]
        except: 
            v_b[v] = 0
    return v_b

def collect_overlapping_partitions(args, fits_df):
    dfs = []
    for idx, row in fits_df.iterrows():
        sub, ses, run = row[['sub', 'ses', 'run']]
        g, state = row[['graph', 'state']]
        modes = row['modes']
        M = len(row['bs'])

        for idx_mode, mode in enumerate(modes):
            # plausibility
            pls = int(args.num_samples*(mode.get_M() / M))
            for s in range(pls):
                # sample <pls> partiitions from mode
                he_b = mode.sample_partition()
                v_b = he_to_v(args, g, state, he_b) 

                df = pd.DataFrame({
                    'sub':[sub],
                    'ses':[ses],
                    'run':[run],
                    'mode':[idx_mode],
                    'sample':[s],
                    'he_b':[he_b],
                    'v_b':[v_b],
                })
                dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)
    
    pmode = gt.ModeClusterState(dfs['v_b'].to_list())
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return dfs, pmode

# =================

def nested_partitions(b, state):
    state = state.copy(bs=b)
    bs = []
    for l, bl in enumerate(b):
        bl_ = np.array(state.project_level(l).get_state().a)
        bs.append(bl_)
        if len(np.unique(bl_)) == 1: break
    return bs
    
def get_nested_mode_partitions(args, M, pmode, g, state):
    mode_df = []
    for idx, mode in enumerate(pmode.get_modes()):
        b = nested_partitions(mode.get_max_nested(), state)
        df = pd.DataFrame({
            'mode':[mode],
            'w':[mode.get_M()/M],
            'sigma':[mode.posterior_cdev()],
            'partition':[b],
            'marginal':[mode.get_marginal(g)],
        })
        mode_df.append(df)
    mode_df = pd.concat(mode_df).reset_index(drop=True)
    return mode_df

def get_mode_partitions(args, M, pmode, g, state):
    mode_df = []
    for idx, mode in tqdm(enumerate(pmode.get_modes())):
        b = mode.get_max(g)
        b_c, sigma_c = gt.partition_overlap_center(list(mode.get_partitions().values()))
        df = pd.DataFrame({
            'mode':[mode],
            'w':[mode.get_M()/M],
            'sigma':[mode.posterior_cdev()],
            'partition':[b],
            'center_partition':[b_c],
            'center_sigma':[sigma_c],
            'marginal':[mode.get_marginal(g)],
        })
        mode_df.append(df)
    mode_df = pd.concat(mode_df).reset_index(drop=True)
    return mode_df

def rescale(X):
    X /= np.expand_dims(np.sum(X, axis=-1), axis=-1)
    X = np.nan_to_num(X)
    X = np.round(X, decimals=3)
    return X

def vertex_overlapping_partition(row, g, state):
    he_b = row['he_b']
    v_b = np.zeros((g.num_vertices(), np.max(np.unique(he_b))+1))
    for v, hes in zip(g.iter_vertices(), state.half_edges):
        blocks, counts = np.unique(he_b[hes], return_counts=True)
        v_b[v, blocks] = counts
    v_b = rescale(v_b)
    return v_b

def get_overlapping_mode_partitions(args, M, pmode, fits_df, dfs):
    mode_df = []
    for idx_mode, mode in enumerate(pmode.get_modes()):
        # find a partition closest to all partitions in the mode
        b_hat, sigma = gt.partition_overlap_center(list(mode.get_partitions().values()))

        # find partition(s) from the collection closest to `b_hat`
        po = lambda x: gt.partition_overlap(x, b_hat)
        dists = dfs['v_b'].apply(po).to_numpy()
        row = dfs.iloc[np.where(dists == np.max(dists))[0]].iloc[0]

        # corresponding graph for the above partition from the collection
        fits_row = fits_df[fits_df['sub'] == row['sub']][fits_df['ses'] == row['ses']]
        g, state = fits_row[['graph', 'state']].iloc[0]

        v_b = vertex_overlapping_partition(row, g, state)

        df = pd.DataFrame({
            'mode':[mode],
            'w':[mode.get_M() / M],
            'sigma':[sigma],
            'partition':[v_b],
            'marginal':[None],
        })
        mode_df.append(df)
    mode_df = pd.concat(mode_df).reset_index(drop=True)
    return mode_df

# =================

def concatenate(in_files, out_file):
    try:
        os.remove(out_file)
    except:
        pass

    tcat = afni.TCat()
    tcat.inputs.in_files = in_files
    tcat.inputs.out_file = out_file
    tcat.inputs.rlt = ''
    tcat.cmdline 
    tcat.run()

    for file in in_files:
        try:
            os.remove(file)
        except:
            pass
    return None

def nested_partition_to_nifti(args, idx_mode, X):
    in_files = []
    parcels = parcels_img.numpy()
    for idx_level, x in enumerate(X):
        x_img = np.zeros_like(parcels)
        for idx, roi in enumerate(roi_labels):
            x_img += (parcels == roi) * (x[idx]+1)
        
        file = f'{NII_path}/group/sbm-{args.dc}-{args.sbm}_mode-{idx_mode}_level-{idx_level}_desc-partition.nii.gz'
        parcels_img.new_image_like(x_img).to_filename(file)
        in_files.append(file)
        
    out_file = f'{NII_path}/group/sbm-{args.dc}-{args.sbm}_mode-{idx_mode}_desc-partition.nii.gz'
    concatenate(in_files, out_file)
    return None

def partition_to_nifti(args, idx_mode, x):
    parcels = parcels_img.numpy()
    x_img = np.zeros_like(parcels)
    for idx, roi in enumerate(roi_labels):
        x_img += (parcels == roi) * (x[idx]+1)
    
    parcels_img.new_image_like(x_img).to_filename(
        f'{NII_path}/group/sbm-{args.dc}-{args.sbm}_mode-{idx_mode}_desc-partition.nii.gz'
    )

def overlapping_partition_to_nifti(args, idx_mode, X):
    in_files = []
    parcels = parcels_img.numpy()
    for idx_group, x in enumerate(X.T):
        x_img = np.zeros_like(parcels)
        for idx, roi in enumerate(roi_labels):
            x_img += (parcels == roi) * (x[idx])
        
        file = f'{NII_path}/group/sbm-{args.dc}-{args.sbm}_mode-{idx_mode}_group-{idx_group}_desc-partition.nii.gz'
        parcels_img.new_image_like(x_img).to_filename(file)
        in_files.append(file)
    
    out_file = f'{NII_path}/group/sbm-{args.dc}-{args.sbm}_mode-{idx_mode}_desc-partition.nii.gz'

    concatenate(in_files, out_file)
    return None

# =================

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
    args.num_samples = int(sys.argv[10]) #5000
    args.SEED = int(sys.argv[11])
    
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
    
    
    print(f'{SBM_path}/*/sbm-{args.dc}-{args.sbm}*')
    files = glob.glob(f'{SBM_path}/*/sbm-{args.dc}-{args.sbm}*')
    fits_df = collect_sbm_fits(args, files)
    # fits_df.head()
    
    g = fits_df.iloc[9]['graph']
    state = fits_df.iloc[9]['state']
    if args.sbm in ['h']:
        dfs, pmode = collect_nested_partitions(args, fits_df)
        mode_df = get_nested_mode_partitions(args, len(dfs['b'].to_list()), pmode, g, state)
    elif args.sbm in ['a', 'd']:
        dfs, pmode = collect_partitions(args, fits_df)
        mode_df = get_mode_partitions(args, len(dfs['b'].to_list()), pmode, g, state)
    elif args.sbm in ['o']:
        dfs, pmode = collect_overlapping_partitions(args, fits_df)
        mode_df = get_overlapping_mode_partitions(args, len(dfs['v_b'].to_list()), pmode, fits_df, dfs)

    out_file = f'sbm-{args.dc}-{args.sbm}_samples-{args.num_samples}_desc-modes.npy'
    with open(f'{NPY_path}/{out_file}', 'wb') as f:
        pickle.dump(
            [mode_df],
            f
        )
    print(f'{NPY_path}/{out_file}')
    
    
    if args.sbm in ['h']:
        for idx_mode, row in mode_df.iterrows():
            nested_partition_to_nifti(args, idx_mode, row['partition'])
    elif args.sbm in ['a', 'd']:
        for idx_mode, row in mode_df.iterrows():
            partition_to_nifti(args, idx_mode, row['partition'])
    elif args.sbm in ['o']:
        for idx_mode, row in mode_df.iterrows():
            overlapping_partition_to_nifti(args, idx_mode, row['partition'])
    return None


# =========================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()