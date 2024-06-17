from algorithms.training import train
import numpy as np
import os

def main(number):
    L = 5        #system size N=L^2
    bond = 64    #bond dimension (for DMRG)
    epsilon = 0  #noise level
    alpha = 0.9  #trainig/validation split


    try: os.makedirs('results/{}'.format(L))
    except:pass
    try: os.makedirs('results/{}/{}'.format(L,epsilon))
    except:pass
    try: os.makedirs('results/{}/{}/{}'.format(L,epsilon,alpha))
    except:pass

    path = 'results/{}/{}/{}'.format(L,epsilon,alpha)


    dims = [L,L]
    train(dims, bond, path, alpha, epsilon, number)
    return None

if __name__ == '__main__':
    number = 0
    for number in range(5):
        _ = main(number)
