import numpy as np
import scipy as sp 
import copy
from itertools import product
from tqdm import tqdm
from scipy import stats
import networkx as nx

import helpers.functions.plotting_utils as plot_utils

def make_dicts(args):
    dicts = {
        f"subj{idx_subj:02}": {
            f"sigma{idx_sigma:02}": {
                f"run{idx_run:02}": None
                for idx_run in range(args.num_runs)
            }
            for idx_sigma in range(args.num_sigmas)
        } 
        for idx_subj in range(args.num_subjs)
    }
    return dicts

def get_average_fcs(args, fcs) -> tuple[dict[str, dict[str, None]], dict[str, dict[str, None]]]:
    fcs_avg = {
        f"subj{idx_subj:02}": {
            f"sigma{idx_sigma:02}": None
            for idx_sigma in range(args.num_sigmas)
        } 
        for idx_subj in range(args.num_subjs)
    }

    fcs_std = copy.deepcopy(fcs_avg)

    for (idx_subj, idx_sigma) in tqdm(
        list(product(range(args.num_subjs), range(args.num_sigmas)))
    ):
        fcs_ = np.stack(
            [
                fcs[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"] 
                for idx_run in range(args.num_runs)
            ],
            axis=0,
        )
        fcs_avg[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"] = np.mean(fcs_, axis=0)
        fcs_std[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"] = np.std(fcs_, axis=0)
    
    return fcs_avg, fcs_std

def compute_fcs(args, out_dicts):
    # functional connectivity
    fcs = make_dicts(args)
    for (idx_subj, idx_sigma) in tqdm(
        list(product(range(args.num_subjs), range(args.num_sigmas)))
    ):
        for idx_run in range(args.num_runs):
            out_dict = out_dicts[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"]
            fc = stats.spearmanr(out_dict['x']).statistic
            fc = np.zeros(shape=[args.num_rois]*2) if type(fc) == float else fc
            np.fill_diagonal(fc, 0.0)
            # fc = np.log(fc)
            fcs[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"] = fc
    return fcs

def fcs_similarity(args, fcs, dist_obj, g_gt):
    dists = make_dicts(args)    
    G1 = nx.from_numpy_array(g_gt) # ground truth graph
    for (idx_subj, idx_sigma) in tqdm(
        list(product(range(args.num_subjs), range(args.num_sigmas)))
    ):
        for idx_run in range(args.num_runs):
            g2 = fcs[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"]
            G2 = nx.from_numpy_array(g2)
            dists[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{idx_run:02}"] = dist_obj.dist(G1, G2)

        dists[f"subj{idx_subj:02}"][f"sigma{idx_sigma-0:02}"] = [
            dists[f"subj{idx_subj:02}"][f"sigma{idx_sigma-0:02}"][f"run{idx_run:02}"] 
            for idx_run in range(args.num_runs)
        ]
    return dists

def calculate_fc_dists(args, networks, fcs, dist_objs, dist_names, log_scales):  
    all_dists = {}
    for dist_obj, dist_name, log_scale in zip(dist_objs, dist_names, log_scales):
        dists = fcs_similarity(
            args, fcs, 
            dist_obj=dist_obj,
            g_gt=networks[0].copy()
        )
        args.log_scale = log_scale
        args.dist_name = dist_name
        plot_utils.plot_fc_dists_hists(args, dists)
        all_dists[dist_name] = dists

    return all_dists

class CosineDistance():
    def __init__(self):
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        """Compute distance between two graphs.

        Values computed as side effects of the distance method can be foun
        in self.results.

        Parameters
        ----------

        G1, G2 (nx.Graph): two graphs.

        Returns
        -----------

        distance (float).

        """
        A1 = nx.to_numpy_array(G1).flatten()
        A2 = nx.to_numpy_array(G2).flatten()
        dist = sp.spatial.distance.cosine(A1, A2)
        self.results['dist'] = dist  # store dist in self.results
        # self.results[..] = ..     # also store other values if needed
        return dist  # return only one value!