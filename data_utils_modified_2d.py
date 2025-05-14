import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, BRICS, rdMolDescriptors, rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MolFromSmiles, FragmentOnBonds, RDConfig, FragmentCatalog
from rdkit.Avalon import pyAvalonTools

from sklearn.metrics import pairwise_distances
from collections import defaultdict
from torch.utils.data import Dataset
from torch_geometric.utils.smiles import from_smiles


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def load_data_from_smiles(x_smiles, labels, add_dummy_node=True, one_hot_formal_charge=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all = [], []

    for smiles, label in zip(x_smiles, labels):
        #try:
        #    mol = MolFromSmiles(smiles)
        try:
            mol = MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
            mol = Chem.RemoveHs(mol)
            afm, adj, edge_attr, dist, com = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj, dist, com])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    return x_all, y_all

from typing import Dict, List, Any

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def featurize_mol(mol, pos, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    assert mol.GetNumAtoms() == pos.shape[0]
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1
    
    ##compute distance matrix with 2d information
    rdDepictor.Compute2DCoords(mol)

    positions = []
    ptable = Chem.GetPeriodicTable()
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    ## additionally add center of mass
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        pos_2d = mol.GetConformer().GetAtomPosition(atom_idx)
        positions.append([pos_2d.x, pos_2d.y])
        atomic_num = atom.GetAtomicNum()
        mass = ptable.GetAtomicWeight(atomic_num)
        x,y,z = pos[atom_idx]
        accum_pos += np.array([x*mass, y*mass, z*mass])
        accum_mass += mass
    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)
    dist_matrix = pairwise_distances(np.array(positions))
    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.zeros((len(edge_attr) + 1, len(edge_attr[0])))
        m[1:, :] = edge_attr
        edge_attr = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return torch.tensor(node_features, dtype=torch.float32), torch.tensor(adj_matrix, dtype=torch.int64), \
              torch.tensor(dist_matrix, dtype=torch.float32), \
              torch.tensor(center_of_mass, dtype=torch.float32), torch.tensor(np.array(positions), dtype=torch.float32)


def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())


    attributes += one_hot_vector(
        atom.GetHybridization(),
        [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
    )

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())
    attributes.append(atom.GetDegree())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)

def smi_to_graph_data(smi, pos):

    atom_data = defaultdict(list)

    mol = Chem.MolFromSmiles(smi)
    num_atoms = mol.GetNumAtoms()
    if num_atoms != pos.shape[0]:
        return None
    else: 
        num_bonds = mol.GetNumBonds()

        afm, adj, dist, com, pos_2d = featurize_mol(mol, pos, add_dummy_node=False, one_hot_formal_charge=True)
        
        element = []
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            atomic_number = atom.GetAtomicNum()
            element.append(atomic_number)
        element = torch.tensor(element, dtype=torch.int64)
        
        atom_data['afm'], atom_data['adj'], atom_data['dist'], atom_data['com'], atom_data['pos'], atom_data['element'], atom_data['pos_2d'] = afm, adj, dist, com, pos, element, pos_2d
        
        return atom_data


def get_mol_fragment_sets_brics(mol):
    num_atoms = mol.GetNumAtoms()
    mol_fragments = BRICS.BreakBRICSBonds(mol)
    mol_fragment_sets = []
    mol_fragments = Chem.GetMolFrags(mol_fragments, asMols=True, fragsMolAtomMapping=mol_fragment_sets)
    mol_fragment_idx_sets = []
    for mol_fragment_set in mol_fragment_sets:
        mol_fragment_idx_set = [atom_idx for atom_idx in mol_fragment_set if atom_idx < num_atoms]
        mol_fragment_idx_set.sort()
        mol_fragment_idx_sets.append(mol_fragment_idx_set)
    brics_bond_list = list(BRICS.FindBRICSBonds(mol, randomizeOrder=False))
    brics_bond_list = [bond for bond, _ in brics_bond_list]
    return mol_fragments, mol_fragment_idx_sets, brics_bond_list

def get_mol_fragment_sets_murcko(mol):
    num_atoms = mol.GetNumAtoms()
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_index = mol.GetSubstructMatch(core)
    # find bonds that are attached to murcko core scaffold
    murcko_bond_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        link_score = 0
        if u in scaffold_index:
            link_score += 1
        if v in scaffold_index:
            link_score += 1
        if link_score == 1:
            murcko_bond_list.append((u, v))

    mol_bond_list = []
    for bond in mol.GetBonds():
        mol_bond_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    # find bond index of murcko core scaffold
    murcko_bond_idx_list = []
    for murcko_bond in murcko_bond_list:
        if murcko_bond in mol_bond_list:
            murcko_bond_idx_list.append(mol_bond_list.index(murcko_bond))
        elif murcko_bond[::-1] in mol_bond_list:
            murcko_bond_idx_list.append(mol_bond_list.index(murcko_bond[::-1]))
            
    if len(murcko_bond_idx_list) == 0:
        # no bond to fragment on
        # print(f'smiles : {Chem.MolToSmiles(mol)} has no fragments')
        mol_fragments  = [mol]
        mol_fragment_idx_sets = [tuple(range(mol.GetNumAtoms()))]
    else:
        # fragment on bonds that are attached to murcko core scaffold
        mol_fragments = Chem.FragmentOnBonds(mol, murcko_bond_idx_list)
        mol_fragment_sets = []
        mol_fragments = Chem.GetMolFrags(mol_fragments, asMols=True, fragsMolAtomMapping=mol_fragment_sets)
        mol_fragment_idx_sets = []
        for mol_fragment_set in mol_fragment_sets:
            mol_fragment_idx_set = [atom_idx for atom_idx in mol_fragment_set if atom_idx < num_atoms]
            mol_fragment_idx_set.sort()
            mol_fragment_idx_sets.append(mol_fragment_idx_set)
    return mol_fragments, mol_fragment_idx_sets, murcko_bond_list

def return_fg_without_ca_i_wash(fg_with_ca_i, fg_without_ca_i):
    # the fragment genereated from smarts would have a redundant carbon, here to remove the redundant carbon
    fg_without_ca_i_wash = []
    for fg_with_ca in fg_with_ca_i:
        for fg_without_ca in fg_without_ca_i:
            if set(fg_without_ca).issubset(set(fg_with_ca)):
                fg_without_ca_i_wash.append(list(fg_without_ca))
    return fg_without_ca_i_wash

def merge_hit_fg_atoms(sorted_all_hit_fg_atoms):
    merged_hit_fg_atoms_list = []
    for hit_fg_atoms in sorted_all_hit_fg_atoms:
        if hit_fg_atoms not in merged_hit_fg_atoms_list:
            if len(merged_hit_fg_atoms_list) == 0:
                merged_hit_fg_atoms_list.append(hit_fg_atoms)
            else:
                hit_fg_atoms_set = set(hit_fg_atoms)
                found_subset = False
                for merged_fg_atoms in merged_hit_fg_atoms_list:
                    if hit_fg_atoms_set.issubset(set(merged_fg_atoms)):
                        found_subset = True
                        break
                if not found_subset:
                    merged_hit_fg_atoms_list.append(hit_fg_atoms)
    return merged_hit_fg_atoms_list

def get_mol_fragment_sets_fg(mol):
    num_atoms = mol.GetNumAtoms()
    
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    fg_without_ca_smart = ['[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
                            'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
                            '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
                            '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
                            '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
                            '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
                            '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
                            '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
                            '[C;D4]([C;D1])([C;D1])-[C;D1]',
                            '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1', '[O;D2]-[C;D2]-[C;D1;H3]',
                            '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]']
    fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]
    fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(fparams.GetNumFuncGroups())]
    fg_name_list = [fg.GetProp('_Name') for fg in fg_with_ca_list]
    
    hit_fg_atoms_list = []
    hit_fg_name_list = []
    all_hit_fg_atoms_list = []
    for i in range(len(fg_with_ca_list)):
        fg_with_ca_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_ca_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_ca_i_wash = return_fg_without_ca_i_wash(fg_with_ca_i, fg_without_ca_i)
        if len(fg_without_ca_i_wash) > 0:
            hit_fg_atoms_list.append(fg_without_ca_i_wash)
            hit_fg_name_list.append(fg_name_list[i])
            all_hit_fg_atoms_list += fg_without_ca_i_wash

    sorted_all_hit_fg_atoms = sorted(all_hit_fg_atoms_list, key=lambda x: len(x), reverse=True)
    
    merged_hit_fg_atoms_list = merge_hit_fg_atoms(sorted_all_hit_fg_atoms)
    
    hit_fg_atm_washed = []
    hit_fg_name_washed = []
    for j in range(len(hit_fg_atoms_list)):
        hit_fg_atm_washed_j = []
        for fg in hit_fg_atoms_list[j]:
            if fg in merged_hit_fg_atoms_list:
                hit_fg_atm_washed_j.append(fg)
        if len(hit_fg_atm_washed_j) > 0:
            hit_fg_atm_washed.append(hit_fg_atm_washed_j)
            hit_fg_name_washed.append(hit_fg_name_list[j])
    
    # make mol_fragment_sets
    mol_fragment_sets = []
    fg_atm_list = []
    for hit_fg_atm_washed_j in hit_fg_atm_washed:
        for mol_fragment_set in hit_fg_atm_washed_j:
            mol_fragment_sets.append(tuple(mol_fragment_set))
            fg_atm_list.extend(mol_fragment_set)
    not_fg_atom_idx = [i for i in range(num_atoms) if i not in fg_atm_list]
    if len(not_fg_atom_idx) != 0:
        mol_fragment_sets.append(tuple(not_fg_atom_idx))
    
    mol_fragment_sets = sorted(mol_fragment_sets, key=lambda x: len(x), reverse=True)

    # make frag_bond_list and frag_bond_idx_list
    frag_bond_list = []
    frag_bond_idx_list = []
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        for frag_idx, mol_fragment_set in enumerate(mol_fragment_sets):
            if begin_atom_idx in mol_fragment_set:
                begin_atom_idx_frag_idx = frag_idx
            if end_atom_idx in mol_fragment_set:
                end_atom_idx_frag_idx = frag_idx
        if begin_atom_idx_frag_idx != end_atom_idx_frag_idx:
            frag_bond_list.append((begin_atom_idx, end_atom_idx))
            frag_bond_idx_list.append(bond_idx)
    
    # fragment on frag_bond_list
    if len(frag_bond_idx_list) != 0:
        mol_fragments = Chem.FragmentOnBonds(mol, frag_bond_idx_list)
        mol_fragment_sets = []
        mol_fragments = Chem.GetMolFrags(mol_fragments, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mol_fragment_sets)
        new_mol_fragment_sets = []
        for mol_fragment_set in mol_fragment_sets:
            mol_fragment_set = [atom_idx for atom_idx in mol_fragment_set if atom_idx < num_atoms]
            new_mol_fragment_sets.append(mol_fragment_set)
        return mol_fragments, new_mol_fragment_sets, frag_bond_list
    else:
        # print(f'smiles: {Chem.MolToSmiles(mol)} has no fragments')
        mol_fragments = [mol]
        mol_fragment_sets = [tuple(range(mol.GetNumAtoms()))]
        return mol_fragments, mol_fragment_sets, frag_bond_list
    

def get_mol_fragment_sets(mol, fragmentation_method):
    if fragmentation_method == 'brics':
        return get_mol_fragment_sets_brics(mol)
    elif fragmentation_method == 'murcko':
        return get_mol_fragment_sets_murcko(mol)
    elif fragmentation_method == 'fg':
        return get_mol_fragment_sets_fg(mol)

def get_hyper_node_feature_sum(data, mol_fragment_set):
    hyper_node_feature = np.array(data.x)[mol_fragment_set].sum(axis=0)
    return hyper_node_feature.tolist()

def get_frag_edges(mol_fragment_sets, frag_bond_list):
    frag_edges = []
    for frag_bond in frag_bond_list:
        frag_edge = []
        for bond_atom_idx in frag_bond:
            for frag_idx, mol_fragment_set in enumerate(mol_fragment_sets):
                if bond_atom_idx in mol_fragment_set:
                    frag_edge.append(frag_idx)
                    break
        frag_edges.append(tuple(frag_edge))
        frag_edges.append(tuple(frag_edge[::-1]))
    return frag_edges


def fragment_pos(data, mol_fragment_set):
    ptable = Chem.GetPeriodicTable()
    fragment_pos = np.array(data.pos)[mol_fragment_set]
    fragment_pos_2d = np.array(data.pos_2d)[mol_fragment_set]

    fragment_atom_weight = [ptable.GetAtomicWeight(int(atomic_number)) for atomic_number in np.array(data.element)[mol_fragment_set]]
    fragment_atom_weight = np.array(fragment_atom_weight).reshape(-1,1)
    
    frag_sum_weight = np.sum(fragment_atom_weight)
    fragment_center_of_mass = np.sum(fragment_pos * fragment_atom_weight, axis=0) / frag_sum_weight
    fragment_center_of_mass_2d = np.sum(fragment_pos_2d * fragment_atom_weight, axis=0) / frag_sum_weight
    
    return frag_sum_weight, fragment_center_of_mass.tolist(), fragment_center_of_mass_2d.tolist()


def mol2_frag_graph(mol, atom_graph_data, fragmentation_method):
    if mol is None:
        return None
    if len(mol.GetAtoms()) == 0:
        return None
    
    data = defaultdict(list)
    mol_fragments, mol_fragment_sets, frag_bond_list = get_mol_fragment_sets(mol, fragmentation_method)

    frag_weights, frag_pos_dist = [], []
    for mol_fragment in mol_fragment_sets:
        frag_feature = get_hyper_node_feature_sum(atom_graph_data, np.array(mol_fragment))
        frag_weight, frag_pos, frag_pos_2d = fragment_pos(atom_graph_data, np.array(mol_fragment))
        data['frag_features'].append(frag_feature)
        data['frag_coor'].append(frag_pos)
        data['frag_coor_2d'].append(frag_pos_2d)
        frag_weights.append(frag_weight)
    
    center_of_mass = np.sum(np.array(data['frag_coor']) * np.array(frag_weights).reshape(-1,1), axis=0) / np.sum(frag_weights)

    distance = pairwise_distances(data['frag_coor_2d'])

    adjacency_matrix = atom_graph_data.edge_index
    edge_indices = torch.nonzero(torch.triu(adjacency_matrix, diagonal=1), as_tuple=True)
    edge_index = torch.stack(edge_indices, dim=0)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    edge_index = edge_index.t()
    
    if len(frag_bond_list) != 0:
        frag_edges = get_frag_edges(mol_fragment_sets, frag_bond_list)
    else:
        data['frag_edges'] = np.zeros((0, 2), dtype=np.int64)
        
        
    data['frag_features'] = torch.tensor(data['frag_features'], dtype=torch.float32)
    data['frag_edges'] = torch.tensor(frag_edges, dtype=torch.int64)
    data['frag_dist'] = torch.tensor(distance, dtype=torch.float32)
    data['frag_pos'] = torch.tensor(data['frag_coor'], dtype=torch.float32)
    data['num_frags'] = torch.tensor(len(mol_fragment_sets), dtype=torch.int64)
    data['frag_center_of_mass'] = torch.tensor(center_of_mass, dtype=torch.float32)
    return data
