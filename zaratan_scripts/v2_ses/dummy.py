import sys
import os
import pandas as pd
import numpy as np
import graph_tool.all as gt

def setup_args():
    class ARGS():
        pass
    args = ARGS()
    
    args.all_graphs_file = sys.argv[1]
    args.graph_file = int(sys.argv[2]) # file of graph, line number (index) in all_graphs.txt
    args.sbm = sys.argv[3] # a d o h
    args.dc = sys.argv[4] == 'True' # degree corrected?
    args.wait = int(sys.argv[5]) # 24,000
    args.force_niter = int(sys.argv[6]) # 40,000
    args.B = int(sys.argv[7]) # 1
    args.SEED = int(sys.argv[8]) # random seed
    
    return args

def load_graph(args, ):
    with open(f'{args.all_graphs_file}', 'r') as f:
        all_graphs = f.readlines()
        for idx, file in enumerate(all_graphs):
            all_graphs[idx] = file[:-1] if file[-1] == '\n' else file
    
    args.graph_file = all_graphs[args.graph_file]
    g = gt.load_graph(args.graph_file)
    return g

def create_state(args, ):
    state_df = pd.DataFrame(columns=['a', 'd', 'o', 'h'],)
    state_df.loc['state'] = [
        gt.PPBlockState, gt.BlockState, 
        gt.OverlapBlockState, gt.NestedBlockState,
    ]
    state_df.loc['state_args'] = [
        dict(), dict(deg_corr=args.dc, B=args.B), 
        dict(deg_corr=args.dc, B=args.B), dict(deg_corr=args.dc, B=args.B),
    ]
    state, state_args = state_df[args.sbm]
    return state, state_args

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['a'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

def mcmc_eq(args, g, state):
    print('inside mcmc_eq')
    bs = [] # partitions
    Bs = np.zeros(g.num_vertices() + 1) # number of blocks
    Bes = [] # number of effective blocks
    dls = [] # description length history
    def collect_partitions(s):
        bs.append(s.b.a.copy())
        B = s.get_nonempty_B()
        Bs[B] += 1
        Bes.append(s.get_Be())
        dls.append(s.entropy())
        
    gt.mcmc_equilibrate(
        state, 
        wait=args.wait, 
        force_niter=args.force_niter,
        mcmc_args=dict(niter=args.niter), 
        callback=collect_partitions,
    )
    # return [None]*5
    return state, bs, Bs, Bes, dls

def main():
    print(f'hello, inside main!')
    args = setup_args()

    g = load_graph(args)
    
    fs = args.graph_file.split('/')
    ROI_RESULTS_path = '/'.join(fs[:-2])
    SBM_path = (
        f'{ROI_RESULTS_path}/model-fits'
        f'/{"_".join(fs[-1].split("_")[:-1])}' 
        f'/{sbm_name(args)}/B-{args.B}'
    )
    os.system(f'mkdir -p {SBM_path}')

    args.num_draws = int((1/2) * args.force_niter) # remove burn-in period
    args.niter = 10
    
    state, state_args = create_state(args)
    state = state(g, **state_args)
    
    # state = gt.minimize_blockmodel_dl(g, state=state, state_args=state_args,)

    atts = [f'{att}: {getattr(args, att)}' for att in dir(args) if not att.startswith('__')]
    print(*atts, sep='\n')
    print(
        *[
            f'graph: {g}',
            f'state: {state}',
            f'state_args: {state_args}',
        ],
        sep='\n',
    )

    state, bs, Bs, Bes, dls = mcmc_eq(args, g, state)
    print(
        *[
            f'Bes: {Bes[:10]}',
        ],
        sep='\n',
    )

if __name__ == "__main__":
    main()