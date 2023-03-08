import os
from tqdm import tqdm, trange
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

def batch_tanimoto(fp, fps):
    inter = np.logical_and(fp, fps)
    union = np.logical_or(fp, fps)
    sims = inter.sum(-1)/union.sum(-1)
    return sims

def batch_tanimoto_faster(fp, fps, fp_sum):
    inter = np.logical_and(fp, fps)
    inter_sum = inter.sum(-1)
    sims = inter_sum/(fp_sum + fp.sum() - inter_sum)
    return sims


def get_clusters(fps, start_index, stop_index, cutoff):
    seen = set()
    clusters = []
    fp_sum = fps.sum(-1)

    tanimoto = batch_tanimoto_faster
    for cur_index in trange(start_index, stop_index):
        fp = fps[cur_index]
        if cur_index in seen:
            continue
        sims = tanimoto(fp, fps, fp_sum)
        indexes = np.argwhere(sims > cutoff)
        cluster = frozenset(indexes[:,0].tolist())
        seen = seen.union(cluster)
        clusters.append(cluster)
    return clusters

class SingleCluster:

    def __init__(self, fps, cutoff, num_threads):
        self.fps = fps
        self.size = len(fps)//num_threads
        self.cutoff = cutoff

    def __call__(self, index):
        start = index*self.size
        end = (index+1)*self.size
        return get_clusters(self.fps, start, end, self.cutoff)

def merge(all_clusters):
    while True:
        to_add = set()
        to_remove = set()

        item2clusters = defaultdict(set)
        for cluster in all_clusters:
            for item in cluster:
                item2clusters[item].add(cluster)

        seen = set()
        for item, clusters in item2clusters.items():
            if item in seen: continue
            if len(clusters) == 1: continue
            merged = frozenset().union(*clusters)
            to_remove = to_remove.union(clusters)
            to_add.add(merged)
            seen = seen.union(merged)

        if len(to_add) == 0:
            assert len(to_remove) == 0
            break
        all_clusters = all_clusters.difference(to_remove).union(to_add)
    return all_clusters

def parellel_cluster(fps, cutoff, num_threads):
    single_cluster = SingleCluster(fps, cutoff, num_threads)
    args = list(range(num_threads))
    with Pool(num_threads) as p:
        all_clusters = set()
        for clusters in p.map(single_cluster, args):
            all_clusters = all_clusters.union(clusters)
                    
    return merge(all_clusters)

def cluster_hits(df, clusters):
    rows = []
    for cluster in tqdm(clusters):
        row = df.loc[cluster].sort_values("score", ascending=False).reset_index(drop=True).loc[0]
        rows.append(row)
    new_df = pd.DataFrame(rows)
    return new_df.sort_values("score", ascending=False).reset_index(drop=True) 

def main(cutoff=0.7, num_threads=32):
    df = pd.read_csv("all_hits.csv")

    fp_file = "hit_fingerprints.pkl"

    if os.path.exists(fp_file):
        with open(fp_file, "rb") as f:
            fps = pickle.load(f)
    else:
        fps = []
        for smi in tqdm(df.smiles):
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=124)
            fps.append(np.array(fp, dtype=bool))
        fps = np.asarray(fps)
        with open(fp_file, "wb") as f:
            pickle.dump(fps, f)

    all_clusters = parellel_cluster(fps[:50000], cutoff, num_threads)
    new_df = cluster_hits(df, all_clusters)
    new_df.to_csv("clustered_hits.csv", index=False)

if __name__ == "__main__":
    main()


