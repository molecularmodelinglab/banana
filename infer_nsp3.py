#!/usr/bin/env python

import argparse
import torch
import sys
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob

from terrace.comp_node import Input
from terrace.batch import make_batch_td, DataLoader

from datasets.csv_dataset import CSVDataset
from datasets.data_types import IsActiveData
from models.make_model import make_model


def inference():

    device = "cpu"

    cfg = OmegaConf.load("configs/classification.yaml")
    in_node = Input(make_batch_td(IsActiveData.get_type_data(cfg)))
    
    model = make_model(cfg, in_node)
    model.load_state_dict(torch.load("data/banana_final.pt"))
    model = model.to(device)
    model.eval()

    index = int(sys.argv[1])
    csv_file = glob("/work/users/m/i/mixarcid/44M_diversity_set.csv_*")[index]
    out_file = f"out_{index}.txt"
    pdb_file = "nsp3_pocket.pdb"

    print(f"Running inference on {csv_file}. Outputting to {out_file}")

    dataset = CSVDataset(cfg, csv_file, pdb_file)
    dataloader = DataLoader(dataset, batch_size=1,
                            num_workers=1, pin_memory=True,
                            shuffle=False)

    with open(out_file, "w") as f:
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            output = model(batch).cpu().numpy()
            for out, valid in zip(output, batch.is_active.cpu()):
                f.write(f"{out},{valid.item()}\n")

if __name__ == "__main__":
    with torch.no_grad():
        inference()

