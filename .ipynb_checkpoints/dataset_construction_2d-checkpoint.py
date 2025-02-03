import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_dense_adj
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import sys
sys.path.append('../')
from data_utils_modified_2d import *


class Fragment(Data):
    def __init__(self, frag_graph, edge_index, y):
        super(Fragment, self).__init__()
        self.x = frag_graph['frag_features']
        self.edge_index = edge_index
#        self.frag_edge_attr = frag_graph['frag_edge_attr']
        self.dist = frag_graph['frag_dist']
        self.pos = frag_graph['frag_pos']
        self.center_of_mass = frag_graph['frag_center_of_mass']
        self.atom_hyper_node_idx = frag_graph['atom_hyper_node_idx']
        self.num_frags = frag_graph['num_frags']
        self.y = y

    def __str__(self):
        return f'x: {self.x}, edge_index: {self.edge_index}, dist: {self.dist}, pos: {self.pos}, center_of_mass: {self.center_of_mass}, atom_hyper_node_idx: {self.atom_hyper_node_idx}, num_frags: {self.num_frags}, y: {self.y}'

class Atom(Data):
    def __init__(self, atom_data, y):
        super(Atom, self).__init__()
        self.x = atom_data['afm']
        self.edge_index = atom_data['adj']
        self.edge_attr = atom_data['edge_attr']
        self.dist = atom_data['dist']
        self.pos = atom_data['pos']
        self.center_of_mass = atom_data['com']
        self.element = atom_data['element']
        self.y = y

    def __str__(self):
        return f'x: {self.x}, edge_index: {self.edge_index}, edge_attr: {self.edge_attr}, dist: {self.dist}, pos: {self.pos}, center_of_mass: {self.center_of_mass}, element: {self.element}, y: {self.y}'

def pad_common(data_list, type):
    if type == 'atom': 
        max_num_atoms = max([len(atom_data.x) for atom_data in data_list])
#        max_num_edges = max([atom_data.edge_attr.size(0) for atom_data in data_list])
        
        for atom_data in data_list:

            node_feature = torch.zeros(max_num_atoms, atom_data.x.size(1))
            node_feature[:atom_data.x.size(0), :] = atom_data.x
            atom_data.x = node_feature

            edge = torch.zeros(max_num_atoms, max_num_atoms)
            edge[:atom_data.edge_index.size(0), :atom_data.edge_index.size(1)] = atom_data.edge_index
            atom_data.edge_index = edge

            dist = torch.zeros(max_num_atoms, max_num_atoms)
            dist[:atom_data.dist.size(0), :atom_data.dist.size(1)] = atom_data.dist
            atom_data.dist = dist
    
    if type == 'frag':
        max_num_frags = max([frag_data.num_frags for frag_data in data_list])

        for frag_data in data_list:
            try:     
                frag_feature = torch.zeros(max_num_frags, frag_data.x.size(1))
                frag_feature[:frag_data.x.size(0), :] = frag_data.x
                frag_data.x = frag_feature

                frag_edge_index = torch.zeros(max_num_frags, max_num_frags)
                frag_edge_index[:frag_data.edge_index.size(0), :frag_data.edge_index.size(1)] = frag_data.edge_index
                frag_data.edge_index = frag_edge_index

                frag_dist = torch.zeros(max_num_frags, max_num_frags)
                frag_dist[:frag_data.dist.size(0), :frag_data.dist.size(1)] = frag_data.dist
                frag_data.dist = frag_dist
            except:
                print(frag_data)

            

    return data_list

  
def normalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Invalid SMILES string')
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

        if Chem.MolFromSmiles(canonical_smiles) is None:
            raise ValueError('Canonical SMILES is invalid')
    except Exception as e:
        print(f'Error in normalization: {e}')
        canonical_smiles = None
    return canonical_smiles



class MultiFragDataset(Dataset):
    def __init__(self, df, smiles_col, target_col, split, normalize_coordinates):
        super(MultiFragDataset, self).__init__()
        self.df = df
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.split = split
#        self.fragmentation_methods = fragment
        self.normalize_coordinates = normalize_coordinates
        self.idx_list, self.atom_data_list, self.brics_atom_list, self.murcko_atom_list, self.fg_atom_list = self.get_data_list()
        self.pad_features()
        

    def pad_features(self):
        self.atom_data_list = pad_common(self.atom_data_list, 'atom')
        print('brics')
        self.brics_atom_list = pad_common(self.brics_atom_list, 'frag')
        print('murcko')
        self.murcko_atom_list = pad_common(self.murcko_atom_list, 'frag')
        print('fg')
        self.fg_atom_list = pad_common(self.fg_atom_list, 'frag')
        return self.atom_data_list, self.brics_atom_list, self.murcko_atom_list, self.fg_atom_list


    def get_data_list(self):
        print("Generating dataset...")
        atom_data_list = []
        frag_list = [[], [], []]
        idx_list = []
        idx = 0

        for row in tqdm(self.df.to_records()):
            smiles = row[self.smiles_col]
            canonical_smiles = normalize_smiles(smiles)
            if canonical_smiles is None:
                continue
            target_value = torch.tensor(row[self.target_col], dtype=torch.float)            
            pos_data = row['mol']
            if self.normalize_coordinates: 
                pos_data = pos_data - pos_data.mean(dim=0)
                mol = Chem.MolFromSmiles(canonical_smiles)
                atom_graph_data = smi_to_graph_data(canonical_smiles, pos_data)
                if atom_graph_data is None:
                    continue

                atom_data = Data(
                            x = atom_graph_data['afm'],
                            edge_index = atom_graph_data['adj'],
                            edge_attr = atom_graph_data['edge_attr'],
                            dist = atom_graph_data['dist'],
                            pos = atom_graph_data['pos'],
                            center_of_mass = atom_graph_data['com'],
                            element = atom_graph_data['element'],
                            y = target_value.clone().detach()
                        )
                atom_data_list.append(atom_data)

                for i, frag in enumerate(['brics', 'murcko', 'fg']):
                    frag_graph_data = mol2_frag_graph(mol, atom_data, frag)
                    if frag_graph_data['frag_edges'].shape[0] != 0:
                        edge_index = to_dense_adj(frag_graph_data['frag_edges']).squeeze(0)
                    else:
                        # Handle the case where there's only one atom (no edges)
                        edge_index = torch.zeros((1, 1))
                    
                    frag_graph_data = Data(
                            x = frag_graph_data['frag_features'],
                            edge_index = edge_index,
        #                    frag_edge_attr = merged_frag_graph_data['frag_edge_attr'],
                            dist = frag_graph_data['frag_dist'],
                            pos = frag_graph_data['frag_pos'],
                            frag_center_of_mass = frag_graph_data['frag_center_of_mass'],
                            atom_hyper_node_idx = frag_graph_data['atom_hyper_node_idx'],
                            num_frags = frag_graph_data['num_frags'],
                            y = target_value.clone().detach()
                        )
                    frag_list[i].append(frag_graph_data)

                idx_list.append(idx)

            idx += 1

        return idx_list, atom_data_list, frag_list[0], frag_list[1], frag_list[2]     

    def __len__(self):
        return len(self.atom_data_list)
    
    def __getitem__(self, idx):
#        padded = self.pad_features()
        return self.atom_data_list[idx], self.brics_atom_list[idx], self.murcko_atom_list[idx], self.fg_atom_list[idx]

class MultiFragDataset_w_fp(Dataset):
    def __init__(self, df, smiles_col, target_col, split, normalize_coordinates, args):
        super(MultiFragDataset_w_fp, self).__init__()
        self.df = df
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.split = split
#        self.fragmentation_methods = fragment
        self.normalize_coordinates = normalize_coordinates
        self.args = args
        self.idx_list, self.atom_data_list, self.brics_atom_list, self.murcko_atom_list, self.fg_atom_list = self.get_data_list()
        self.pad_features()
        

    def pad_features(self):
        self.atom_data_list = pad_common(self.atom_data_list, 'atom')
        print('brics')
        self.brics_atom_list = pad_common(self.brics_atom_list, 'frag')
        print('murcko')
        self.murcko_atom_list = pad_common(self.murcko_atom_list, 'frag')
        print('fg')
        self.fg_atom_list = pad_common(self.fg_atom_list, 'frag')
        return self.atom_data_list, self.brics_atom_list, self.murcko_atom_list, self.fg_atom_list


    def get_data_list(self):
        print("Generating dataset...")
        atom_data_list = []
        frag_list = [[], [], []]
        idx_list = []
        idx = 0

        for row in tqdm(self.df.to_records()):
            smiles = row[self.smiles_col]
            canonical_smiles = normalize_smiles(smiles)
            if canonical_smiles is None:
                continue
            target_value = torch.tensor(row[self.target_col], dtype=torch.float)            
            pos_data = row['mol']
            if self.normalize_coordinates: 
                pos_data = pos_data - pos_data.mean(dim=0)
                mol = Chem.MolFromSmiles(canonical_smiles)
                atom_graph_data = smi_to_graph_data(canonical_smiles, pos_data)
                fingerprint = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.args.ecfp_radius, nBits=self.args.ecfp_dim), dtype=int)
                if atom_graph_data is None:
                    continue

                atom_data = Data(
                            x = atom_graph_data['afm'],
                            edge_index = atom_graph_data['adj'],
                            edge_attr = atom_graph_data['edge_attr'],
                            dist = atom_graph_data['dist'],
                            pos = atom_graph_data['pos'],
                            pos_2d = atom_graph_data['pos_2d'],
                            center_of_mass = atom_graph_data['com'],
                            element = atom_graph_data['element'],
                            fp = torch.tensor(fingerprint, dtype=torch.float).unsqueeze(0),
                            y = target_value.clone().detach()
                        )
                atom_data_list.append(atom_data)

                for i, frag in enumerate(['brics', 'murcko', 'fg']):
                    frag_graph_data = mol2_frag_graph(mol, atom_data, frag)
                    if frag_graph_data['frag_edges'].shape[0] != 0:
                        edge_index = to_dense_adj(frag_graph_data['frag_edges']).squeeze(0)
                    else:
                        # Handle the case where there's only one atom (no edges)
                        edge_index = torch.zeros((1, 1))
                    
                    frag_graph_data = Data(
                            x = frag_graph_data['frag_features'],
                            edge_index = edge_index,
        #                    frag_edge_attr = merged_frag_graph_data['frag_edge_attr'],
                            dist = frag_graph_data['frag_dist'],
                            pos = frag_graph_data['frag_pos'],
                            frag_center_of_mass = frag_graph_data['frag_center_of_mass'],
                            atom_hyper_node_idx = frag_graph_data['atom_hyper_node_idx'],
                            num_frags = frag_graph_data['num_frags'],
                            y = target_value.clone().detach()
                        )
                    frag_list[i].append(frag_graph_data)

                idx_list.append(idx)

            idx += 1

        return idx_list, atom_data_list, frag_list[0], frag_list[1], frag_list[2]     

    def __len__(self):
        return len(self.atom_data_list)
    
    def __getitem__(self, idx):
#        padded = self.pad_features()
        return self.atom_data_list[idx], self.brics_atom_list[idx], self.murcko_atom_list[idx], self.fg_atom_list[idx]
