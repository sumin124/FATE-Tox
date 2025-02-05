import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
from models.common import GaussianSmearing, MLP



class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='relu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10. ** 2
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.r_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian, fixed_offset=False)
        
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian*2, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        else: 
            self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
        #print(self.edge_mlp)

    def forward(self, h, x, edge_index, edge_attr):
        dst, src = edge_index  
        hi, hj = h[dst], h[src]
        rel_x = x[dst] - x[src]
        x_l2 = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        edge_attr = edge_attr.float()
        hi = hi.float()
        hj = hj.float()

        mij = self.edge_mlp(torch.cat([edge_attr, self.r_expansion(x_l2), hi, hj], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        output = self.node_mlp(torch.cat([mi, h], -1))
        if self.update_x:
            xi, xj = x[dst], x[src]
            delta_x = scatter_sum((xi - xj) * self.x_mlp(mij), dst, dim=0, dim_size=x.shape[0])
            
            delta_x = delta_x / xi.shape[0] # stabilize by dividing to num of atoms
            x = x + delta_x
            
        return output, x


class EnEquiEncoder(nn.Module):
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, num_r_gaussian, k=32, cutoff=10.0,
                 update_x=True, act_fn='relu', norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian, fixed_offset=False)
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, node_attr, pos, batch):
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        x = pos

        for interaction in self.net:
            delta_h, x = interaction(h, x, edge_index, edge_attr)
            output = h + delta_h
        return output
