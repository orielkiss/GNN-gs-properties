import numpy as np
import torch
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from algorithms.model import *
from algorithms.data_preparation import *
import tqdm

def train(dims, bond, path, alpha, epsilon = 0, number = 1):

    ####
    a = 1
    OBC = True
    learn_energy = False
    lr = 10**-3
    epoch = 1201
    dropout=0.05

    #########################################
    N = dims[0]*dims[1]
    #Load data

    dataset = load_data(dims, bond, epsilon, alpha, a, learn_energy, OBC)
    l = len(dataset)
    print(dataset[0].edge_attribute.shape)
    # print(dataset[0].x)
    edge_dim = 1
    if dataset[0].weight_attribute is not None:
        edge_dim = dataset[0].edge_attribute.shape[1]

    dataset_train = dataset[:int(alpha*l)]
    dataset_test = dataset[int(alpha*l):]

    criterion = torch.nn.MSELoss()

    batch_size = int(alpha*l) # we dont do minibatch (small number of data and is a bit unstable)


    loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)
    #########################################
    ############### model preparation #########

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    layers = [[N,128,128,64,64,32,32,N,N],[N,32,64,32,N]]

    if learn_energy:
        layers[1][-1] = 1
    assert layers[0][-1]==layers[1][0]

    model = GCN(layers, edge_dim, dropout, N, batch_size, learn_energy).to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.995)
    model.train()
    model.double()

    test_loss = []
    train_loss = []

    best_val = 100
    best_param = None
    ##### training #####
    for epoch in tqdm.tqdm(range(epoch)):
        model.batch_size = batch_size
        for batch in loader:

            optimizer.zero_grad()
            out = model(batch)

            if learn_energy:
                out = out.reshape(batch_size,-1)
                out = torch.mean(out,axis=-1)


            loss = criterion(out,batch.y)
            loss.backward()
            if epoch>500:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()
            l = loss.cpu().detach().numpy()
            train_loss.append(l)

            # stop if convergence is achieved
            if len(train_loss)>100:
                if abs(train_loss[-100]-l)<10**-6: break

        if scheduler.get_last_lr()[0]>10**-5 and epoch%50 ==0:

            scheduler.step()

        # validation
        model.batch_size = len(dataset_test)
        for test in loader_test:

            out = model(test)
            if learn_energy:
                out = out.reshape(model.batch_size,-1)
                out = torch.mean(out,axis=-1)

            l = criterion(out,test.y).cpu().detach().numpy()
            test_loss.append(l)

            if l<best_val:
                torch.save(model.state_dict(), path+'/model_{}.pt'.format(number))
                np.save(path+'/predictions{}.npy'.format(number),out.cpu().detach().numpy())
                np.save(path+'/validation_data{}.npy'.format(number),test.y.cpu().detach().numpy())



        if epoch%50==0:
            # print(loss, test_loss[-1])
            steps = np.arange(0,len(train_loss),int(len(train_loss)/len(test_loss)))
            np.save(path+'/train_loss{}.npy'.format(number),np.array(train_loss) )
            np.save(path+'/test_loss{}.npy'.format(number),np.array(test_loss) )
            np.save(path+'/steps{}.npy'.format(number),np.array(steps))


    # plotting loss and validation examples 
    plt.figure()
    plt.plot(steps,test_loss,'.',color='darkviolet',label='validation')
    plt.plot(train_loss,color='green',label = 'training')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.savefig(path+'/training{}.png'.format(number))

    if not learn_energy:
        for i in range(1):
            model.batch_size = 1
            pred = model(dataset_test[i]).cpu().detach().numpy()[0,...]

            plt.figure()
            plt.imshow(pred,vmin=-1,vmax=1)
            plt.savefig(path+'/pred_ex{}.png'.format(number))
            plt.figure()

            plt.imshow(dataset_test[i].y.cpu().reshape(N,N),vmin=-1,vmax=1)
            plt.savefig(path+'/data_ex{}.png'.format(number))
