import numpy as np
import pandas as pd
from glob import glob
from os.path import join, basename

proj_path = '/home/govindas/lab-data/aba'
reg_path = '/data/KABA_STORAGE/ABA_paper2/stimtimes_ABApaper2'
# ts_path = join(proj_path,'cleaned_timeseries_esc69')
# ts_glob = join(ts_path,'ABA???_resids_REML.1D')
cond_cols = ['play_highR','play_highT','play_lowR','play_lowT',
             'feedback_NotRec_highR','feedback_NotRec_highT','feedback_NotRec_lowT','feedback_NotRec_lowR',
             'feedback_Rec_highR','feedback_Rec_highT','feedback_Rec_lowT','feedback_Rec_lowR',]

num_trs_per_run = 410

def to_TR(t):
    return int(round(float(t)/1.25))

def zscore(ts):
    ts = ts - np.mean(ts,axis=0)
    ts = ts / np.std(ts,axis=0)
    return ts

def get_dataset(ts_dir='cleaned_timeseries_esc69'):

    ts_path = join(proj_path,ts_dir)
    ts_glob = join(ts_path,'ABA???_resids_REML.1D')
    print(ts_glob)

    df_cols = ['pid','rid','timeseries','input_ts']
    df_cols += cond_cols
    df_data = {col:[] for col in df_cols}

    ts_files = sorted(glob(ts_glob))
    print(f"{len(ts_files)} subjects in the dataset")

    for ts_file in ts_files:

        ts_filename = basename(ts_file)
        pid = ts_filename.split('_')[0]

        ts_allruns = np.loadtxt(ts_file)
        num_rois, num_runs = ts_allruns.shape[0], ts_allruns.shape[1]//num_trs_per_run
        if num_runs * num_trs_per_run != ts_allruns.shape[1]: 
            print(f"{pid} is faulty")
            continue
        ts_allruns = np.split(ts_allruns, num_runs, axis=1)

        for rid in range(num_runs):
            
            ts = ts_allruns[rid].T
            ts = zscore(ts)

            input_ts = []
            for col in cond_cols:

                if "play" in col:
                    dur=8
                else:
                    dur=1

                cond_reg_file = join(reg_path,pid,"regs",f"{pid}_{col}.txt")
                with open(cond_reg_file,'r') as f:
                    reg = np.zeros(num_trs_per_run)
                    times = [to_TR(i.strip()) for i in f.readlines()[rid].split('\t') if '*' not in i]
                    df_data[col].append(times)
                    for t in times:
                        if dur>1:
                            reg[t-dur:t+1]
                        elif dur==1:
                            reg[t] = 1
                    input_ts.append(reg)
            input_ts = np.stack(input_ts).T

            df_data['pid'].append(pid)
            df_data['rid'].append(rid)
            df_data['timeseries'].append(ts)
            df_data['input_ts'].append(input_ts)

    df = pd.DataFrame(df_data)
    return df