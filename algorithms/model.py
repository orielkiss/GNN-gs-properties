import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GraphConv, SimpleConv, SAGEConv, global_mean_pool,GENConv, NNConv
from torch_geometric.nn.pool import global_max_pool



class GCN(torch.nn.Module):
    """
    Graph Convolutional Network
    Args:
       - layers: list of sizes of the hidden GrapConv layers, and hidden fully connected layers
       - N (int): number of spins
       - edge_dim (int): dimension of the couplings
       - dropout (int): dropout probabilities
       - batch_size: number of data point per optimization step
       - learn_energy (bool): flag if only the energy is to learn
    """
    def __init__(self,layers, edge_dim, dropout, N, batch_size, learn_energy):
        super().__init__()

        self.conv = []
        self.linear = []
        self.dropout = dropout
        self.N =  N
        self.batch_size = batch_size,
        self.learn_energy = learn_energy


        for l in range(len(layers[0])-1):

            if edge_dim >1:
                conv_ = GENConv(layers[0][l], layers[0][l+1], edge_dim = 2,  aggr='sum',
                learn_t = True, learn_p = True, num_layers = 2)
            else:
                conv_ = GraphConv(layers[0][l], layers[0][l+1])
            self.conv.append(conv_)


        self.conv = nn.ModuleList(self.conv)

        for l in range(len(layers[1])-1):
            linear_ = torch.nn.Linear(in_features = layers[1][l], out_features = layers[1][l+1])
            self.linear.append(linear_)

        self.linear = nn.ModuleList(self.linear)

        self.last = torch.nn.Linear(self.N, out_features = 1) #only used if a single output is learned



    def forward(self, data):
        x = data.x

        for conv in self.conv:
            x = conv(x, data.edge_index,  data.edge_attribute).relu()
            x = F.dropout(x, p=self.dropout)


        for _,lin in enumerate(self.linear):
            x = lin(x)
            if _<len(self.linear)-1:
                x = x.relu()


        if self.learn_energy:

            x = x.view(self.batch_size,self.N)
            x = self.last(x)
            return x.sigmoid()

        # we need to format the output to be symmetric images
        return self.dot(x.tanh())

    def dot(self,out):

        """
        return a symmetric image with ones on the diagonal
        """
        out = out.reshape(self.batch_size,self.N, self.N)
        g = torch.nn.functional.normalize(out.view(out.size(0)* out.size(1) , out.size(2)), p=2, dim=1)
        out = g.view(*out.size())
        out = torch.einsum('bij,bkj->bik',out,out)

        return out
