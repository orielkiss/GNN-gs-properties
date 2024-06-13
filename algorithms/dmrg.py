import tenpy
import numpy as np
import warnings

from tenpy.networks.site import SpinSite, SpinHalfSite
from tenpy.models.lattice import Chain, Square, TrivialLattice
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, exact_diag
from tenpy.algorithms import tebd
from tenpy.algorithms.dmrg import SubspaceExpansion, DensityMatrixMixer
from tenpy.models.spins import SpinChain, SpinModel

warnings.filterwarnings("ignore")


class XXZ_custom(SpinModel):
    r"""Spin-1/2 XXZ chain with Sz conservation.

    The Hamiltonian reads:

    .. math ::
        H = \sum_i \mathtt{Jxx}/2 (S^{+}_i S^{-}_{i+1} + S^{-}_i S^{+}_{i+1})
                 + \mathtt{Jz} S^z_i S^z_{i+1} \\
            - \sum_i \mathtt{hz} S^z_i

    All parameters are collected in a single dictionary `model_params` and read out with
    :func:`~tenpy.tools.params.get_parameter`.

    Parameters
    ----------
    N : int
        number of spins
    J : float | array
        Couplings as defined for the Hamiltonian above.

    """
    def __init__(self, params):
        # 0) read out/set default parameters
        name = "XXZ"
        J = params['J']

        Lx,Ly = params['dims']
        hz  = params['hz']
        alpha = params['alpha']


        site = SpinHalfSite()
        # 4) lattice
        bc = 'open'
        bc_MPS = 'finite'
        site = SpinSite(S=0.5, conserve=None)  # You can adjust S (spin) and conserve as needed

        lattice = Chain(Lx*Ly, site, bc=bc, bc_MPS=bc_MPS)

        CouplingModel.__init__(self, lattice)
        links = generate_links_rectangular_graph_open(Lx,Ly)
        # links = create_edges_list(Lx,Ly)

        # add XXZ coupling on the links
        for _,c in enumerate(links):

            Jp = 2*J[0,_]
            Jm = 0
            Jz = alpha*J[0,_]

            self.add_coupling_term(Jp, min(c),max(c), 'Sp',  'Sm', plus_hc=True)
            # self.add_coupling_term(Jm, c[0],c[1], 'Sp',  'Sp', plus_hc=True)
            self.add_coupling_term(2*Jz, min(c), max(c), 'Sz',  'Sz', plus_hc=True)

        # add on site external field
        self.add_onsite(hz,0, 'Sz')



        MPOModel.__init__(self, lattice, self.calc_H_MPO())

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

def create_adjacency_list(num_rows, num_columns, edges):
    """
    create adjancy matrix
    """
    adjacency_list = {i: [] for i in range(num_rows * num_columns)}

    for edge in edges:
        v1, v2 = edge
        adjacency_list[v1].append(v2)
        adjacency_list[v2].append(v1)

    # Include the vertex itself in its list of neighbors
    for vertex in adjacency_list:
        adjacency_list[vertex].append(vertex)

    return list(adjacency_list.values())

def inset_indices(Lx,Ly):
    """
    collect indices for inset squares (used to calculate entropies)
    """
    indices = []
    for l in range(1,3):
        square = []
        for i in range(l):
            for j in range(l):
                square.append(i+Lx*j)
        indices.append(square)
    return indices





def run_dmrg(J,hz, dims, bond, alpha):

    N = dims[0]*dims[1]
    edges = create_edges_list(dims[0],dims[1])
    adjacency_list = create_adjacency_list(dims[0], dims[1], edges)

    params = {'dims':dims,'J':J/N,'hz':hz/N, "alpha":alpha}
    model = XXZ_custom(params)

    sites = model.lat.mps_sites()
    all_connections  = []

    for i in range(N):
        for j in range(N):
            all_connections.append((i,j))


    inset = inset_indices(*dims)


    product_state = []
    for i in range(N):
        product_state.append(np.random.choice(["up", "down"]))

    psi = MPS.from_product_state(sites,product_state,"finite")

    energies = []
    ### EXACT DIAG ###
    if N<16:
        # we can check the implementation for small sizes
        ED = exact_diag.ExactDiag(model)
        ED.build_full_H_from_mpo()
        ED.full_diagonalization()
        print('exact gs energy', min(ED.E))



    bond_dimension = [bond]
    for bd in bond_dimension:
        del psi
        psi = MPS.from_product_state(sites,product_state,"finite")

        dmrg_params = {"trunc_params": {"chi_max": bd, "svd_min": 1.e-10}, "mixer": True,'max_sweeps': 500}

        info = dmrg.run(psi, model, dmrg_params)

        print("E = ",info['E'])
        print("max. bond dimension = ",max(psi.chi))

        psi.test_sanity()

        correlatorsX = [4*psi.expectation_value_term([('Sx', i) for i in l]) for l in all_connections]
        correlatorsY = [4*psi.expectation_value_term([('Sy', i) for i in l]) for l in all_connections]
        correlatorsZ = [4*psi.expectation_value_term([('Sz', i) for i in l]) for l in all_connections]

        single = [2*psi.expectation_value_term([('Sz', i)]) for i in range(N)]
        all = list([2**N*psi.expectation_value_term([('Sz', i) for i in range(N)])])

        entropy = list([psi.entanglement_entropy_segment2(n=1,segment = l) for l in inset])

        observables = [info['E']] + correlatorsX + correlatorsY + correlatorsZ
        # observables = [info['E']] + all + entropy + correlatorsZ + single

    return observables

def main():

    dims = (4,4)

    bond = 50
    alpha = 1

    couplings = []
    field = []
    observables = []


    mode = 'rdm'
    number_links = len(generate_links_rectangular_graph_open(dims[0],dims[1]))
    # number_links = len(create_edges_list(dims[0],dims[1]))


    # uniform
    if mode =='uniform':

        for a in np.linspace(-2,2,20):
            for hz in np.linspace(-2,2,20):
                J = a*np.ones((number_links))
                h = hz*np.ones(dims[0]*dims[1])

                obs = run_dmrg(J, h, dims, bond, alpha)
                observables.append(obs)
                couplings.append(J)
                field.append(hz)


    elif mode=='rdm':

        for i in range(150):
            seed = 42+i
            np.random.seed(i)
            J = np.random.uniform(0,1,size=1*number_links).reshape(1,-1)
            h = 0*np.random.uniform(-2,2,size = dims[0]*dims[1])

            obs  = run_dmrg(J,h, dims, bond, alpha)
            observables.append(obs)
            couplings.append(J)
            field.append(h)

    else:
        pass

    np.save('data/{}/{}_couplings_{}.npy'.format(mode,dims[0],bond),np.array(couplings))
    np.save('data/{}/{}_field_{}.npy'.format(mode,dims[0],bond),np.array(field))
    np.save('data/{}/{}_observables_{}_{}.npy'.format(mode,dims[0],alpha,bond),np.array(observables))



if __name__ == '__main__':
    main()
