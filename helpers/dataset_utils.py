import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def get_max_data_trials(args, data_df, subj_idx_list):
    X = {
        0: [], #safe
        1: [], #threat
    } # (label, subj, trial, time, roi)
    for idx_row in tqdm(subj_idx_list):
        subj, ts, targets = data_df.iloc[idx_row]

        for label in args.LABELS:
            x = []
            for region in contiguous_regions(targets == label):
                x.append(ts[region[0]: region[1], args.roi_idxs])
            x = np.stack(x, axis=0)
            X[label].append(x)
    
    return X

# ----------------------------------

def get_columns_design_matrix(design_mat_path):
    raw_cols = open(design_mat_path, 'r').readlines()[3].split('"')[1].split(';')
    raw_cols = [raw_col.strip() for raw_col in raw_cols]

    design_mat = np.loadtxt(design_mat_path)
    design_mat = pd.DataFrame(design_mat, columns=raw_cols)

    raw_cols = [
        raw_col 
        for raw_col in raw_cols 
        if "Run" not in raw_col 
        if "Motion" not in raw_col
    ]
    design_mat = design_mat[raw_cols]

    used_cols = []
    for col, col_data in design_mat.iteritems():
        if col_data.sum() == 0: continue
        used_cols.append(col)
    
    # display(design_mat[used_cols])

    return raw_cols, used_cols, design_mat


def get_columns_response_file(response_file_path):
    raw_cols = open(response_file_path, 'r').readlines()[7].split('"')[1].split(';')
    used_col_idxs = [
        idx 
        for idx, col in enumerate(raw_cols)
        if 'Ftest' not in col
    ]

    response_file = np.loadtxt(response_file_path)[:, used_col_idxs]
    return response_file


def get_cond_ts(args, response_data, label='FNS#'):
    cols = [
        col 
        for col in response_data 
        if label in col 
        if 'r_' not in col
    ]
    cond_ts = np.stack(
        np.split(
            response_data[cols].to_numpy().astype(np.float32).T, 
            len(cols) // args.TRIAL_LEN, 
            axis=0
        ),
        axis=0
    )
    return cond_ts


def get_max_trial_level_responses(args, main_data_dir, subjs):
    X = {}
    for label, _ in enumerate(args.LABEL_NAMES):
        X[label] = []

    for subj in tqdm(subjs):

        data_dir = f"{main_data_dir}/{subj}"

        design_mat_path = f"{data_dir}/{subj}_Main_block_Deconv.x1D"
        response_file_path = f"{data_dir}/{subj}_Main_block_Deconv_bucket.1D"

        raw_cols, used_cols, design_mat = get_columns_design_matrix(design_mat_path)
        response_data = pd.DataFrame(columns=raw_cols)
        response_data[used_cols] = np.loadtxt(response_file_path)[:, 1::2]

        for label, name in zip(args.LABELS, args.LABEL_NAMES):
            X[label].append(
                get_cond_ts(args, response_data, name).astype(np.float32)
            )
    
    return X, raw_cols, used_cols, 


def get_aba_trial_level_responses(args, main_data_dir, subjs):
    X = {}
    for label, _ in enumerate(args.LABEL_NAMES):
        X[label] = []

    for subj in tqdm(subjs):

        data_dir = f"{main_data_dir}/{subj}"

        design_mat_path = f"{data_dir}/{subj}_Main_block_deconv.x1D"
        response_file_path = f"{data_dir}/{subj}_bucket_REML.1D"

        raw_cols, used_cols, design_mat = get_columns_design_matrix(design_mat_path)
        
        response_data = pd.DataFrame(columns=raw_cols)
        response_data[used_cols] = get_columns_response_file(response_file_path)

        for label, name in enumerate(args.LABEL_NAMES):
            X[label].append(
                get_cond_ts(args, response_data, name).astype(np.float32)
            )
    
    return X, raw_cols, used_cols, 

def get_emo2_ts_single_subj(args, subj, motion_df, X):
    # get proximity and time series
    proximity = np.hstack(
        motion_df.loc[motion_df.pid.isin([subj])]['proximity'].to_list()
    ).T
    direction = np.hstack(
        motion_df.loc[motion_df.pid.isin([subj])]['direction'].to_list()
    ).T
    ts = np.loadtxt(
        f"{args.main_data_path}/CON{subj}/CON{subj}_resids_REML.1D"
    ).T

    assert (proximity.shape[0] == ts.shape[0])
    assert (direction.shape[0] == ts.shape[0])
    assert (np.sum(np.isnan(ts)) == 0)

    # censor proximity values 
    censor_TRs = ts[:,0] == 0
    proximity[censor_TRs] = 0.0

    # near misses
    peaks, props = sp.signal.find_peaks(
        proximity, 
        height=args.near_miss_thresh, 
        width=args.near_miss_width
    )
    
    near_miss_windows = np.round(
        np.stack(
            [props['left_ips'], props['right_ips']], 
            axis=-1
            )
    ).astype(int)

    near_misses = np.zeros_like(proximity)
    for idx in range(near_miss_windows.shape[0]):
        near_misses[near_miss_windows[idx, 0] : near_miss_windows[idx, 1]+1] = 1.0
        # +1 because we need the last TR also
    
    # appr and retr segments
    if args.trial_option == 'concat':
        for idx_label, (label, name) in enumerate(zip(args.LABELS, args.LABEL_NAMES)):
            X[idx_label].append(ts[((direction == label) * (near_misses)).astype(bool), :])
    elif args.trial_option == 'trial':
        for idx_label, (label, name) in enumerate(zip(args.LABELS, args.LABEL_NAMES)):
            trials = contiguous_regions((direction == label) * (near_misses))
            ts_list = [
                ts[trial[0]:trial[1], :]
                for trial in trials
            ]
            # contiguous_regions takes care of the last element, so no need to add 1 in the indices.
            X[idx_label].append(ts_list)

    return proximity, peaks, props

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=lambda i: len(seq[i]))[::-1] # [::-1] for descending order

def get_emo2_trial_level_responses(args, motion_df, subjs):
    X = {}
    for label, _ in enumerate(args.LABEL_NAMES):
        X[label] = []

    for subj in tqdm(subjs):
        proximity, peaks, props = get_emo2_ts_single_subj(args, subj, motion_df, X)

    # for idx_label, label in enumerate(args.LABELS):
    #     X[idx_label] = [X[idx_label][subj] for subj in argsort(X[idx_label])[:-4]]
    if args.trial_option == 'concat':
        return X
    elif args.trial_option == 'trial':
        for idx_label, (label, name) in enumerate(zip(args.LABELS, args.LABEL_NAMES)):
            maxTRs = []
            for xs in X[idx_label]:
                maxTRs += [len(x) for x in xs]
            maxTRs = np.max(maxTRs)
            print(f"{name} : maxTRs {maxTRs}")

            for idx_x in np.arange(len(X[idx_label])):
                # collect all trials into an ndarray with nans
                xs = X[idx_label][idx_x]
                xs_ragged = np.empty(shape=(len(xs), maxTRs, args.num_rois))
                xs_ragged.fill(0.0) #.fill(np.nan)
                for trial in np.arange(len(xs)): #num trials
                    x = xs[trial]
                    t, r = x.shape
                    xs_ragged[trial, :t, :] = x
                X[idx_label][idx_x] = xs_ragged
        
        return X

# ----------------------------------

def sim_data_additive_white_noise(args, X, y):
    '''
    white-noise 
    -----------
    1. std's for each roi and each tp
    2. simulated data with i.i.d. normal noise (white noise) around mean time series
    '''
    X_, y_ = [], []
    for label in args.LABELS:
        idx = y[:, 0] == label
        X_ += [np.random.normal(
            loc=np.mean(X[idx], axis=0), 
            scale=args.noise_level*np.std(X[idx], axis=0), 
            size=X[idx].shape
        )]
        y_ += [np.ones(shape=(X[idx].shape[:-1])) * label]

    X_ = np.concatenate(X_, axis=0)
    y_ = np.concatenate(y_, axis=0)

    perm = np.random.permutation(y_.shape[0])
    X_ = X_[perm]
    y_ = y_[perm]

    return X_, y_

# ----------------------------------

'''
elif trial_option == 'mean':
    trials = contiguous_regions((direction == label) * (near_misses))
    ts_list = [
        ts[trial[0]:trial[1], :]
        for trial in trials
    ]
    # contiguous_regions takes care of the last element, so no need to add 1 in the indices.
    X[idx_label].append(ts_list)

# ragged ndarray
num_rois = X[0][0][0].shape[-1]

for idx_label, (label, name) in enumerate(zip(args.LABELS, args.LABEL_NAMES)):
    maxTRs = []
    for xs in X[idx_label]:
        maxTRs += [len(x) for x in xs]
    maxTRs = np.max(maxTRs)
    print(f"{name} : maxTRs {maxTRs}")

    for idx_x in np.arange(len(X[idx_label])):
        # collect all trials into an ndarray with nans
        xs = X[idx_label][idx_x]
        xs_ragged = np.empty(shape=(len(xs), maxTRs, num_rois))
        xs_ragged.fill(np.nan)
        for trial in np.arange(len(xs)): #num trials
            x = xs[trial]
            t, r = x.shape
            if label == args.APPR:
                # near miss at the end of appr, so align to the end
                xs_ragged[trial, -t:, :] = x
            elif label == args.RETR:
                # near miss at the start of retr, so align to the start
                xs_ragged[trial, :t, :] = x
        X[idx_label][idx_x] = xs_ragged
'''