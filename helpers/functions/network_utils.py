import numpy as np
import scipy as sp 
import networkx as nx




def generate_module(args, network=0, version=0):
    name = args.networks[network][version].name
    graph_generator = args.networks[network][version].graph_generator
    params = args.networks[network][version].params
    params_list = [p for p in params if type(p) is not dict]
    params_dict = [p for p in params if type(p) is dict]
    if len(params_list) > 0 and len(params_dict) == 0:
        G = graph_generator(*params_list)
    elif len(params_list) == 0 and len(params_dict) > 0:
        G = graph_generator(**params_dict[0])
    else:
        G = graph_generator(*params_list, **params_dict[0])
    
    if name in ['sparse random']:
        W = G.A
    else:
        W = nx.to_numpy_array(G)
        w = np.random.rand(*W.shape)
        w = (w + w.T) / 2
        W *= w
    
    W = (W + W.T) / 2
    # W = (W > 0).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    weights = args.scale * W

    return weights

def generate_off_diagonal_block(args, idx_blk1, idx_blk2):
    W = sp.sparse.random(
        m=args.num_rois[idx_blk1],
        n=args.num_rois[idx_blk2],
        density=0.25, #(args.density[idx_blk1]+args.density[idx_blk2])/2,
    ).A
    # W = (W > 0).astype(np.float32)
    weights = -args.scale * W
    if idx_blk1 == 0 and idx_blk2 == 2:
        weights *= 0.0
    return weights

def arrange_submatrices(args, blocks):
    rows = []
    for idx_blk1 in range(len(blocks)):
        rows.append(np.hstack(blocks[idx_blk1]))
    mat = np.vstack(rows)
    return mat

def generate_connectivity_matrix(args, network=0):
    # generate connections between two set of rois
    blocks = [
        [
            None 
            for _ in range(args.num_modules)
        ]
        for _ in range(args.num_modules)
    ]
    for idx_blk1 in range(len(blocks)):
        for idx_blk2 in range(len(blocks[idx_blk1])):
            if idx_blk1 == idx_blk2:
                # intra-group excitatory connections
                blocks[idx_blk1][idx_blk2] = generate_module(args, network=network, version=idx_blk1)
            elif idx_blk2 > idx_blk1:
                # inter-group inhibitory connections
                blocks[idx_blk1][idx_blk2] = generate_off_diagonal_block(args, idx_blk1, idx_blk2)
            elif idx_blk2 < idx_blk1:
                blocks[idx_blk1][idx_blk2] = blocks[idx_blk2][idx_blk1].T
    # return blocks
    # combine all blocks
    weights = arrange_submatrices(
        args, 
        blocks
    )
    roi_labels = []
    for idx_blk in range(len(blocks)):
        # roi labels
        label = args.group_label[idx_blk]
        rls = [f"{label}_{roi+1:02}" for roi in range(args.num_rois[idx_blk])]
        roi_labels += rls

    return (
        weights, 
        roi_labels,
    )

'''
idx_network = +1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'sparse random'
    module.graph_generator = sp.sparse.rand
    module.params = [{'m':args.num_rois[idx_module], 'n':args.num_rois[idx_module], 'density':args.density[idx_module]}]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'watts strogatz'
    module.graph_generator = nx.watts_strogatz_graph
    module.params = [{'n':args.num_rois[idx_module], 'k':5, 'p':0.75, 'seed':int(args.SEEDS[idx_module])}]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'barabasi albert'
    module.graph_generator = nx.extended_barabasi_albert_graph
    module.params = [{'n':args.num_rois[idx_module], 'm':5, 'p':0.05, 'q':0.5, 'seed':int(args.SEEDS[idx_module])}]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'powerlaw'
    module.graph_generator = nx.powerlaw_cluster_graph
    module.params = [{'n':args.num_rois[idx_module], 'm':5, 'p':0.5, 'seed':int(args.SEEDS[idx_module])}]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'tree'
    module.graph_generator = nx.random_powerlaw_tree
    module.params = [{'n':args.num_rois[idx_module], 'gamma':3, 'seed':int(args.SEEDS[idx_module]), 'tries':10000}]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'rand.geometric'
    module.graph_generator = nx.soft_random_geometric_graph
    module.params = [{'n':args.num_rois[idx_module], 'radius':0.25, 'dim':2, 'pos':None, 'p_dist':None, 'seed':int(args.SEEDS[idx_module])}]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'gauss.partition'
    module.graph_generator = nx.gaussian_random_partition_graph
    module.params = [{'n':args.num_rois[idx_module], 's':3, 'v':2, 'p_in':0.5, 'p_out':0.2, 'directed':False, 'seed':int(args.SEEDS[idx_module])}]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'comp.mult.part.'
    module.graph_generator = nx.complete_multipartite_graph
    module.params = args.module_parts[idx_module]
    args.networks[idx_network][idx_module] = module

idx_network += 1
for idx_module in range(args.num_modules):
    module = ARGS()
    module.name = 'stoch.blk.mdl.'
    module.graph_generator = nx.stochastic_block_model
    module.params = [{'sizes':args.module_parts[idx_module], 'p':args.module_ps[idx_module], 'seed':int(args.SEEDS[idx_module])}] 
    args.networks[idx_network][idx_module] = module
'''