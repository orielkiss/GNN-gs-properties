from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


def generate_links_rectangular_graph_open(Lx, Ly):
    """
    generate all the links for a rectangular grid with OBC
    """
    links = []
    for i in range(Lx):
        for j in range(Ly):
            site = i * Ly + j
            if i < Lx - 1:  # Right neighbor
                right_neighbor = (i + 1) * Ly + j
                links.append((site, right_neighbor))
            if j < Ly - 1:  # Upper neighbor
                upper_neighbor = i * Ly + (j + 1)
                links.append((site, upper_neighbor))
    return links

def create_edges_list(Lx, Ly):
    """
    generate all the links for a rectangular grid with PBC
    """
    edges = []

    # Horizontal edges
    for row in range(Lx):
        for column in range(Ly):
            vertex = row * Lx + column
            # Right neighbor
            right_neighbor = row * Lx + (column + 1) % Lx
            edges.append((vertex, right_neighbor))
            # Bottom neighbor
            bottom_neighbor = ((row + 1) % Ly) * Lx + column
            edges.append((vertex, bottom_neighbor))

    return edges

def add_noise(data, epsilon):
    """
    simulate measurement errors on a quantum computer (following binomial distribution)
    """

    if epsilon ==0 :
        # no noise
        return data
    data_noisy = []

    shots = int(1/epsilon**2)

    shape = data.shape
    data = data.reshape(-1)
    for x in data:
        p = (1+x)/2
        p = min(p,1.0)
        p = max(p,-1.0)
        b = np.mean(np.random.binomial(1, p=p, size = shots))
        data_noisy.append(2*b-1)
    return np.array(data_noisy).reshape(shape)


def load_data(dims, bond, epsilon = 0, alpha = 0.8, a = 1, learn_energy = False, OBC = True):
    """
    data preparation function
    Input:
     - dims (tuple of int): dimension of the lattice
     - bond (int): bond dimension of the DMRG protocol
     - epsilon (float): level of data corruption due to measurment errors
     - a (float) : scaling factor between the x and y coupling component
     - learn_energy (bool): flag to learn the energy instead of the correlators
     - OBC (bool): flag to use open boundary condition

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # load the data
    observables = np.load('data/rdm/{}_observables_{}_{}.npy'.format(dims[0],a,bond))
    coupling    = np.load('data/rdm/{}_couplings_{}.npy'.format(dims[0],bond))
    field       = np.load('data/rdm/{}_field_{}.npy'.format(dims[0],bond))

    # if no external fiedl, we set it to 1 since this is an input to the graph neural network
    if np.sum(np.absolute(field))==0:
        field = np.ones_like(field)

    # wfs = np.load('data/rdm/{}_wfs_{}_{}.npy'.format(dims[0],a,8))


    N = dims[0]*dims[1] #number of spins
    l = int(np.shape(observables)[1]/3)


    edge_index = torch.tensor(generate_links_rectangular_graph_open(*dims))  #create edge of the 2d lattice
    if not OBC:
        edge_index = torch.tensor(create_edges_list(*dims))
    print(len(edge_index))
    # extract the energy and scale
    energy = observables[:,0].reshape(-1,1)
    scaler = MinMaxScaler()
    energy = scaler.fit_transform(energy)
    ######

    # extract the correlators
    # correlators = 1/3*np.sum(np.array([np.array(observables[:,1+n*l:1+(n+1)*l]) for n in range(3)]),axis=0).reshape(-1,N,N)
    correlators = observables[:,4:4+N**2].reshape(-1,N,N)
    OOTC = np.zeros((observables.shape[0],N, N))

    _ = 0
    for i in range(N):
        for j in range(N):
            OOTC[:,i,j] = observables[:,4+_] #- observables[:,4+N**2+i]*observables[:,4+N**2+j]
            _+=1

    # correlators = OOTC
    I, J = torch.triu_indices(N, N, offset=1)

   # build the graph datatset
    dataset = []
    for _ in range(field.shape[0]):
        c = correlators[_,...]

        if _< int(alpha*field.shape[0]):
            # add only noise on training data
            c = add_noise(correlators[_,...],epsilon)

        y = torch.tensor(c).reshape(1,N,N).to(device)
        if learn_energy:
             y = torch.tensor(energy[_,0].reshape(1,1)).to(device)
             y = torch.tensor(observables[_,3].reshape(1,1)).to(device)/2

        if len(np.shape(coupling[_,...]))==2:
            # we have to make the difference between one and two dimensional coupling
            data = Data(
                x = torch.diag(torch.tensor(field[_,:],dtype=torch.double)).to(device),
                edge_index  = edge_index.t().contiguous().to(device),
                edge_attribute = torch.tensor(coupling[_,...]).to(device),
                y = y
                       )
        else:
            data = Data(
                x = torch.diag(torch.tensor(field[_,:],dtype=torch.double)).to(device),
                edge_index  = edge_index.t().contiguous().to(device),
                edge_weight = torch.tensor(coupling[_,...]).to(device).T,
                y = y
                       )
        dataset.append(data)

    return dataset
