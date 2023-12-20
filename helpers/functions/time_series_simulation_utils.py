import numpy as np
import scipy as sp 
import copy
from itertools import product
from tqdm import tqdm

def GAM(t, p, q):
    '''
    HRF filter
    '''
    return (t/(p*q))**p * np.exp(p - t/q)


def to_python_vars(out_dict):
    for k, v in out_dict.items():
        out_dict[k] = np.array(v.tomemoryview().tolist())

    out_dict['t'] = out_dict['t'].squeeze()
    for k, v in out_dict.items():
        if k == 't': continue
        out_dict[k] = out_dict[k].T

    return out_dict
    
def simulate(eng, model_path, model, in_dict):
    eng.cd(model_path)
    out_dict = model(in_dict, nargout=1)
    out_dict = to_python_vars(out_dict)

    return out_dict
        
def get_in_dicts_out_dicts(args):
    in_dicts = {
        f"subj{idx_subj:02}": {
            f"sigma{idx_sigma:02}": {
                f"run{idx_run:02}": {}
                for idx_run in range(args.num_runs)
            }
            for idx_sigma in range(args.num_sigmas)
        } 
        for idx_subj in range(args.num_subjs)
    }

    out_dicts = copy.deepcopy(in_dicts)

    return in_dicts, out_dicts

def simulate_time_series(args, in_dict, networks, eng, model_path, model,n=1):
    in_dicts, out_dicts = get_in_dicts_out_dicts(args)

    # time series simulation
    for (idx_subj, idx_sigma) in tqdm(
        list(product(range(args.num_subjs), range(args.num_sigmas)))
    ):
        for idx_run in range(args.num_runs):
            in_dict['Kij'] = networks[idx_subj]
            in_dict['sigma'] = args.sigmas[idx_sigma]
            in_dict['randn'] = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(n*args.num_rois, 100*(args.tspan[-1] - args.tspan[0]),),
            )
            in_dicts[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"] = copy.deepcopy(in_dict)
            
            # sp.io.savemat(
            #     f"{bdmodels_dir}/in_dict_vdP_SDE_B.mat",
            #     in_dict,
            # )

            out_dict = simulate(eng, model_path, model, in_dict)

            out_dicts[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"] = copy.deepcopy(out_dict)
    return in_dicts, out_dicts


def convolve_with_hrf(args, out_dicts):
    for (idx_subj, idx_sigma) in tqdm(
        list(product(range(args.num_subjs), range(args.num_sigmas)))
    ):
        for idx_run in range(args.num_runs):
            out_dict = copy.deepcopy(out_dicts[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"])
            t, x = list(out_dict.values())
            h = GAM(t, args.p, args.q)
            h /= h.sum()
            for idx_roi in np.arange(args.num_rois):
                x[:, idx_roi] = np.convolve(h, x[:, idx_roi], mode='full')[:x.shape[0]]
            out_dict['x'] = x
            out_dicts[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"] = copy.deepcopy(out_dict)
    return out_dicts
            
