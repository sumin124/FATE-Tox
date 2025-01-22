from models.egnn import EnEquiEncoder
from models.common import ShiftedSoftplus
from models.transformer import make_model

import torch.nn as nn
import torch
from torch_scatter import scatter


def get_encoder(args):
    net = EnEquiEncoder(
        num_layers=args.GNN_layers,
        edge_feat_dim=0,
        hidden_dim=args.cdim,
        num_r_gaussian=args.num_r_gaussian,
        act_fn='relu',
        norm=False,
        update_x=True,
        k=args.knn,
        cutoff=10.0,
    )
    return net

class transformer_egnn_branch(nn.Module):
    def __init__(self, args, output_dim=1, tasks=1):
        super(transformer_egnn_branch, self).__init__()        
        self.output_dim = output_dim
        self.tasks = tasks
        self.device = args.device

        MAT_params = {
            'd_model': args.MAT_hdim,
            'N': args.MAT_layers,
            'h': args.MAT_heads,
            'N_dense': 1,
            'lambda_attention': args.lambda_attention, 
            'lambda_distance': args.lambda_distance,
            'leaky_relu_slope': 0.1, 
            'dense_output_nonlinearity': 'relu', 
            'distance_matrix_kernel': 'exp', 
            'dropout': args.dropout,
            'aggregation_type': args.mat_agg
        }

        self.node_embedding = nn.Linear(args.d_atom, args.MAT_hdim)

        self.encoder = nn.ModuleList()
        for _ in range(args.num_layers):
            self.encoder.append(make_model(**MAT_params))
            self.encoder.append(nn.Sequential(
                    nn.Linear(args.MAT_hdim, args.cdim),
#                    nn.BatchNorm1d(args.hdim),
                    nn.ReLU()
                ))
            self.encoder.append(get_encoder(args))
            self.encoder.append(nn.Sequential(
                nn.Linear(args.cdim, args.hdim),
                nn.ReLU()
            ))

    def forward(self,batch):
        pos_ctx=batch.pos
        batch_ctx=batch.batch

        init_atom_embedding = batch.x.to(self.device)
        atom_embedding = self.node_embedding(init_atom_embedding) # convert 33 (num feat) -> 64 dimension
        
        for i,layer in enumerate(self.encoder):
            if i%4==0:
                atom_embedding, batch_mask = self.mat_encode(layer, batch, atom_embedding)#batch.ligand_atom_feature_full.float()
            elif i%4==1:
                atom_embedding = layer(atom_embedding)
            elif i%4==2:
                batch_mask = batch_mask.view(-1)
                atom_embedding=layer(node_attr=atom_embedding,
                      pos=pos_ctx,
                      batch=batch_ctx[batch_mask])
            elif i%4==3:
                atom_embedding = layer(atom_embedding)
            else:
                raise ValueError(i)
            atom_embedding = atom_embedding.to(batch.x.device)
        pre_out = scatter(atom_embedding, index=batch_ctx[batch_mask], dim=0, reduce='sum')  # (N, F)
        pre_out = pre_out.to(self.device)

        return pre_out

    def from_pretrained(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def mat_encode(self, layer, batch, atom_embedding):        
        adjacency_matrix, node_features, distance_matrix = batch.edge_index, batch.x, batch.dist
        adjacency_matrix = adjacency_matrix.reshape(batch.num_graphs, adjacency_matrix.shape[0], -1)
        node_features = node_features.reshape(batch.num_graphs, -1, node_features.shape[-1])
        distance_matrix = distance_matrix.reshape(batch.num_graphs, -1, distance_matrix.shape[-1])

        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        
        adjacency_matrix=adjacency_matrix.to(self.device)
        node_features=node_features.to(self.device)
        distance_matrix=distance_matrix.to(self.device)

        # apply atom_embedding on masked indices
        unmasked_atom_embedding = torch.zeros(node_features.shape[0],node_features.shape[1],atom_embedding.shape[-1]).to(self.device)

        if unmasked_atom_embedding[batch_mask].shape[0] == atom_embedding.shape[0]:
            unmasked_atom_embedding[batch_mask] = atom_embedding
            atom_embedding = unmasked_atom_embedding
        else:
            atom_embedding = atom_embedding.reshape(batch.num_graphs, -1, atom_embedding.shape[-1])

        res=layer.encode(atom_embedding, batch_mask, adjacency_matrix, distance_matrix,None)
        
        return res[batch_mask], batch_mask


class fate_tox_model(nn.Module):

    def __init__(self, args, output_dim=1, tasks=1):
        super(fate_tox_model, self).__init__()        
        self.config = config
        self.hidden_dim = args.hdim
        self.output_dim = output_dim
        self.tasks = tasks
        self.device = args.device

        self.atom_encoder = transformer_egnn_branch(args, output_dim, tasks).to(self.device)
        self.frag_encoder = transformer_egnn_branch(args, output_dim, tasks).to(self.device)

        args.murcko_weight = 1 - args.brics_weight - args.fg_weight
        self.brics_weight = nn.Parameter(torch.tensor(args.brics_weight, requires_grad=False, device=self.device))
        self.fg_weight = nn.Parameter(torch.tensor(args.fg_weight, requires_grad=False, device=self.device))
        self.murcko_weight = nn.Parameter(torch.tensor(args.murcko_weight, requires_grad=False, device=self.device))

        self.clf_heads=nn.ModuleList()
        for _ in range(tasks):
            self.clf_heads.append(nn.Sequential(
                nn.Linear(self.hidden_dim*2+args.ecfp_dim, self.hidden_dim),
                nn.Dropout(args.clf_dropout),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(args.clf_dropout),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, self.output_dim),
            ))

    def forward(self, batch):

        atom, brics, murcko, fg = batch
        for data in [atom, brics, murcko, fg]:
            for key in data.keys():
                data[key] = data[key].to(self.device)

        atom_pre_out = self.atom_encoder(atom)
        brics_out = self.frag_encoder(brics)
        murcko_out = self.frag_encoder(murcko)
        fg_out = self.frag_encoder(fg)

        
        frag_pre_out = brics_out * self.brics_weight + murcko_out * self.murcko_weight + fg_out * self.fg_weight
        
        atom_pre_out_w_fp = torch.cat([atom_pre_out, atom['fp']], dim=1)
        
        pre_out = torch.cat([atom_pre_out_w_fp, frag_pre_out], dim=1)
        preds = [head(pre_out) for head in self.clf_heads]

        return torch.cat(preds,dim=1).to(self.device)
