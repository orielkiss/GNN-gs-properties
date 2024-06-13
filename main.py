from algorithms.training import train
import numpy as np
import os

def main(number):
    L = 4
    bond = 64
    epsilon = 0
    alpha = 0.8 # ratio training/validation set

    try: os.makedirs('results/{}'.format(L))
    except:pass
    try: os.makedirs('results/{}/{}'.format(L,epsilon))
    except:pass
    try: os.makedirs('results/{}/{}/{}'.format(L,epsilon,alpha))
    except:pass

    path = 'results/{}/{}/{}'.format(L,epsilon,alpha)


    dims = [L,L]
    train(dims, bond, path, alpha, epsilon, number)

if __name__ == '__main__':
    number = 0
    for number in range(1):
        main(number)
