import numpy as np
import scipy as sp
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

def basis_expansion(X, K, TYPE, NODE):
    '''
     basis_expansion(X,K,TYPE,NODE) generates a multidimensional array of
     basis expansions evaluated on all points of a multivariate time series.

     Parameters
     ------------------
     X:    Matrix containing N time series of M time points.
     K:    Maximum order of the basis expansion.
     TYPE: Type of basis function employed. In this file, we only
           implement expansions up to pairwise interactions. Expansions
           availables are: polynomial (x_{j}^{k}), polynomial_diff
           ((x_{j}-x_{NODE})^{k}), power_series (x_{j}^{k1}*x_{i}^{k2}),
           fourier (sin(k*x_{j}) and cos(k*x_{j})), fourier_diff
           (sin(k*(x_{j}-x_{NODE})) and cos(k*(x_{j}-x_{NODE}))) and RBF (a model
           based on radial basis functions). These functions are shown in
           table I in the main manuscript.
     NODE: Unit on which we are performing the reconstruction. Zero indexed.

     Input type
     ------------------
     X:    double
     K:    integer
     TYPE: string
     NODE: integer

     Output
     ------------------
     Expansion: Multidimensional array of size [K+1,M,N] containing the
     evalation of all k=0,1,...,K basis functions for all M time points and
     all N possible incoming connections. For power_series, (K*K+2) basis
     functions are employed, and for fourier(_diff), 2*(K+1) are employed.

     Example
     ------------------
     basis_expansion(X,4,'power_series',5); generates a multidimensional array
     of size [18,M,N] containing the evaluation of the basis for all M time
     points and all N possible incoming connections.

     Accompanying material to "Model-free inference of direct interactions
     from nonlinear collective dynamics".

     Author: Jose Casadiego
     Date:   May 2017
    '''

    N,M = X.shape
    Expansion = np.zeros((K+1, M, N))

    if TYPE == 'polynomial':
        for n in range(N):
            for k in range(K):
                Expansion[k,:,n] = X[n,:]**k

    elif TYPE == 'polynomial_diff':
        Xi =np.zeros((N,M))
        for m in range(M):
            Xi[:,m] = X[:,m]-X[NODE,m]

        for n in range(N):
            for k in range(K):
                Expansion[k,:,n] = Xi[n,:]**k

    elif TYPE == 'fourier':
        Expansion=np.zeros((2*(K), M,N))
        for n in range(N):
            t = 0
            for k in range(K):
                Expansion[k+t,:,n] = np.sin(k*X[n,:])
                Expansion[k+t+1,:,n] = np.cos(k*X[n,:])
                t += 1

    elif TYPE == 'fourier_diff':
        Expansion=np.zeros((2*(K),M,N))
        Xi = np.zeros((N,M))
        for m in range(M):
            Xi[:,m] = X[:,m] - X[NODE, m]

        for n in range(N):
            t = 0
            for k in range(K):
                Expansion[k+t,:,n] = np.sin(k*Xi[n,:])
                Expansion[k+t+1,:,n] = np.cos(k*Xi[n,:])
                t += 1

    elif TYPE == 'power_series':
        Expansion = np.zeros(((K)*(K), M, N))
        for n in range(N):
            for k1 in range(K):
                for k2 in range(K):
                    for m in range(M):
                        Expansion[((K)*k1)+k2, m, n] = (X[NODE,m]**k1)*(X[n,m]**k2)

    elif TYPE == 'RBF':
        Expansion = np.zeros((K,M,N))
        for n in range(N):
            A = np.vstack((X[n,:], X[NODE,:]))
            for m1 in range(K):
                for m2 in range(M):
                    Expansion[m1,m2, n] = np.sqrt(2.0+np.linalg.norm(A[:,m1]-A[:,m2],2)**2)

    return(Expansion)



def reconstruct_single_node(X, MODEL, ORDER, BASIS, NODE, CONNECTIVITY):
    '''
    returns a ranked list of the inferred incoming connections

    Parameters
    ------------------
    X: Matrix containing N time series of M time points.
    MODEL: Dynamic model employed. This is only used to specify whether the
        time series come from 1D systems like kuramoto1 or 3D systems like
        roessler. Thus, it is not used during the actual reconstruction.
    NODE:  Unit upon the reconstruction takes place. Zero indexed
    BASIS: Type of basis employed. Currently, polynomial, polynomial_diff,
        power_series, fourier, fourier_diff and RBF are supported. For
        more detailed information, please see 'Functions/basis_expansion.m'
        and Table I in the main manuscript.
    ORDER: Number of basis in the expansion.
    CONNECTIVITY: ground truth network

    Output
    ------------------
    list: Sequence of inferred interactions in the order such were detected.
    cost: Fitting cost for all inferred interactions in the order such were
        detected.
    FPR:  False positives rate for the reconstruction.
    TPR:  True positives rate for the reconstruction.
    AUC:  Quality of reconstruction measured in AUC scores.
    '''

    DX = np.diff(X, n=1, axis=-1, prepend=0)
    if MODEL in ('kuramoto', 'kuramoto1', 'kuramoto2'):
        #Transforming data coming from phase oscillators
        X = np.mod(X, 2*np.pi)
    else:
        X = X

    N,M = X.shape

    #Stopping criterium: decrease it to recover longer list of possible links
    th=0.0001

    # Beginning of reconstruction algorithm
    # print('Performing ARNI...')
    # Y[basis, sample, node]
    Y = basis_expansion(X, ORDER, BASIS, NODE)
    nolist = list(range(N))
    llist = []
    cost = []
    b=1
    vec = np.zeros(N,)
    while (nolist and (b==1)):
        #composition of inferred subspaces
        Z = np.array([])
        for n in range(len(llist)):
            Z = np.vstack((Z,Y[:,:,llist[n]])) if Z.size else Y[:,:,llist[n]]

        # projection on remaining composite spaces
        P = np.zeros((len(nolist),2))
        cost_err = np.zeros(len(nolist),)
        for n in range(len(nolist)):
            #composition of a possible spaces
            R = np.vstack((Z, Y[:,:,nolist[n]])) if Z.size else Y[:,:,nolist[n]]
            #error of projection on possible composite space
            # ( A.R=DX)
            RI = np.linalg.pinv(R)
            A = np.dot(DX[NODE,:], RI)
            DX_est = np.dot(A, R)
            DIFF = DX[NODE,:] - DX_est
            P[n,0] = np.std(DIFF) # the uniformity of error
            P[n,1] = int(nolist[n])
            #Fitting cost of possible composite space
            cost_err[n] =  (1/float(M)) * np.linalg.norm(DIFF)
            R = np.array([])

        # break if all candidates equivalent
        if np.std(P[:,0]) < th:
            b=0
            break

        else:
            #Selection of composite space which minimises projection error
            MIN = np.min(P[:,0]) #best score
            block = np.argmin(P[:,0]) #node index of best
            llist.append(int(P[block,1])) # add best node ID to llist
            nolist.remove(int(P[block,1])) # remove best from candidate list
            vec[int(P[block,1])] = MIN # used in ROC curve
            cost.append(cost_err[block]) # record SS Error
    # print('Reconstruction has finished!')

    if not llist:
        # print('WARNING: no predicted regulators - check that NODE abundance varies in the data!')
        AUC = np.nan
        FPR = [np.nan]
        TPR = [np.nan]

    else:
        #load CONNECTIVITY for comparison
        adjacency = CONNECTIVITY
        adjacency[adjacency != 0] = 1

        #adding degradation rate to true adjecency matric of Michaelis-menten systems
        if MODEL == 'michaelis_menten':
            for i in range(N):
                adjacency[i,i] = 1

        # print('Quality of reconstruction:')

        if (np.sum(adjacency[NODE,:]) == 0):
            print('WARNING: no true regulators!')
            AUC = np.nan
            FPR = [np.nan]
            TPR = [np.nan]
        else:
            # Evaluation of results via AUC score
            FPR, TPR, _ = roc_curve(np.abs(adjacency[NODE,:]),
            np.abs(vec),)
            AUC = auc(FPR, TPR)
            FPR = np.insert(FPR,0,0.)
            TPR = np.insert(TPR,0,0.)

        # print(AUC)
    return(llist, cost, FPR, TPR, AUC)

def reconstruct(X, model, order, basis, connectivity):
    reconstructions = []
    for idx_node in tqdm(np.arange(X.shape[0])):
        reconst = reconstruct_single_node(
            X=X, 
            MODEL=model,
            ORDER=order,
            BASIS=basis,
            NODE=idx_node,
            CONNECTIVITY=connectivity,
        )
        reconstructions.append(reconst) # llist, cost, FPR, TPR, AUC
        
    return reconstructions

def get_inferred_network(args, reconstructions):
    W_ = np.zeros(shape=(args.num_rois, args.num_rois))
    for idx_node, reconst in enumerate(reconstructions):
        W_[idx_node, reconst[0]] = 1
    return W_