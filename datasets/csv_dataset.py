
import torch
from torch.utils import data
from rdkit import Chem

from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb
from datasets.data_types import IsActiveData

class CSVDataset(data.Dataset):
    """ Dataset for inference script. Just loads a smi file and a pdb file,
    and rins with it """

    def __init__(self, cfg, csv_file, pdb_file):
        super().__init__()
        self.cfg = cfg
        with open(csv_file, "r") as f:
            self.smiles = [ line.split(",")[0].strip() for line in f ]
        if self.smiles[0] == "smiles":
            self.smiles = self.smiles[1:]
        self.prot_graph = prot_graph_from_pdb(cfg, pdb_file)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        mol = Chem.MolFromSmiles(self.smiles[index])
        active = True
        # very hacky -- us is_active as validity flag
        if mol is None:
            mol = Chem.MolFromSmiles("CCCC")
            print(f"Error loading {self.smiles[index]}")
            active = False
        lig_graph = MolGraph(self.cfg, mol, use_3d=False)
        is_active = torch.tensor(active, dtype=bool)

        return IsActiveData(lig_graph, self.prot_graph, is_active)
