# FATE-Tox

FATE-TOX is a multi-organ molecular toxicity prediction model that leverages both atom-level and three fragment-level graphs to interpret molecules in their diverse existing structures within the human body. The model integrates:

2D molecular graphs processed through a molecule attention transformer to extract atomic and substructural features while explicitly using the knowledge of bond connectivity.
Molecule features refined via an E(3)-equivariant graph neural network (EGNN), with the additional information of 3D spatial coordinates to capture geometric interactions.
The dual branch structure of FATE-Tox allows fragment-based attention mechanisms to enhance interpretability by identifying key toxicophoric substructures.

Multi-task learning (MTL) to jointly predict toxicity endpoints, improving generalization over single-task models.
By incorporating multiple chemical knowledge based fragmentation methods with differing weights across specific endpoints, FATE-TOX outperforms conventional methods in toxicity prediction, demonstrating its effectiveness in structure-aware molecular property analysis.

# Environment 
Install `conda` environment

```
conda env create -f environment.yaml
```

# Usage
- `mol_stage1.py`: Constructing atom-level and fragment-level graphs given a SMILES string of a molecule. 
  **Dataset Construction from _data_df_ (dataframe of SMILES-target toxicity label)
  ```
  train_dataset = MultiFragDataset_w_fp(data_df, args.smiles_col, args.target_col, 'train', args.coor_normalize, args)
  ```

- `main.py`: Run model 
  
  ```
  python main.py
    ```
