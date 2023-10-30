from collections import defaultdict
from glob import glob
import pandas as pd
import random

from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
from common.cache import cache
# from meeko import PDBQTMolecule

from datasets.bigbind_screen import BigBindScreenDataset
from datasets.lit_pcba import LitPcbaDataset
from common.old_routine import get_old_model, old_model_key, get_weight_artifact

class ValModel(ABC):
    """ The whole 'ValModel' thing is just a very hacky way to compare BANANA to gnina
    and/or vina. Will change in future! """

    @abstractmethod
    def get_cache_key(self):
        return

    @abstractmethod
    def get_name(self):
        return

    @abstractmethod
    def __call__(self, x, dataset):
        return

    def to(self, device):
        return self

class OldModel(ValModel):

    def __init__(self, cfg, run, tag):
        self.run = run
        self.model = get_old_model(cfg, run, tag)
        self.key = old_model_key(cfg, run, tag)

    def get_cache_key(self):
        return self.key

    def get_name(self):
        artifact = get_weight_artifact(self.run)
        return f"{self.run.id}_{artifact.version}"

    def __call__(self, x, dataset):
        return self.model(x)

    def to(self, device):
        self.model = self.model.to(device)
        return self

# class VinaModel(ValModel):

#     def __init__(self, cfg):
#         self.dir = cfg.platform.bigbind_docked_dir

#     def get_cache_key(self):
#         return "vina"

#     def get_name(self):
#         return "vina"

#     def __call__(self, batch, dataset):
#         assert isinstance(dataset, BigBindScreenDataset)
#         ret = []
#         for index in batch.index:
#             pdbqt_file = f"{self.dir}/{dataset.split}_screens/{dataset.target}/{index}.pdbqt"
#             if os.path.exists(pdbqt_file):
#                 ret.append(-PDBQTMolecule.from_file(pdbqt_file).score)
#             else:
#                 ret.append(-100)
#         return torch.tensor(ret, dtype=torch.float32, device=batch.index.device)

@cache(lambda cfg, target, dense: (target, dense))
def get_gnina_bigbind_scores(cfg, target, dense):
    dfs = {}
    for file in glob(f"{cfg.platform.bigbind_gnina_dir}/protein_tars/{target}/*_results.csv"):
        rec_pdb = file.split("/")[-1].split("_")[0]
        dfs[rec_pdb] = pd.read_csv(file)

    key = "cnn_affinity_dense" if dense else "cnn_affinity_default"

    all_scores = [ torch.tensor(df[key], dtype=torch.float32) for df in dfs.values() ]
    best_scores = torch.stack(all_scores).amax(0)
    return best_scores

class GninaModel(ValModel):

    def __init__(self, cfg, dense):
        self.cfg = cfg
        self.dense = dense
        self.pcba_scores = None
        self.bigbind_scores = {}

    def get_cache_key(self):
        return "gnina_dense" if self.dense else "gnina"

    def get_name(self):
        return self.get_cache_key()

    def __call__(self, batch, dataset):

        if isinstance(dataset, BigBindScreenDataset):
            if dataset.target not in self.bigbind_scores:
                best_scores = get_gnina_bigbind_scores(self.cfg, dataset.target, self.dense)
                self.bigbind_scores[dataset.target] = best_scores
            
            scores = self.bigbind_scores[dataset.target]
            ret = []
            for index in batch.index:
                ret.append(scores[index])
            return torch.tensor(ret, dtype=torch.float32, device=batch.index.device)

        assert isinstance(dataset, LitPcbaDataset)

        if self.pcba_scores is None:
            self.pcba_scores = {}
            score_file = "./prior_work/lit-pcba_dense-CNNaffinity-mean-then-max.summary" if self.dense else "./prior_work/newdefault_CNNaffinity-max.summary"
            with open(score_file) as f:
                for line in f.readlines():
                    _, score, target, idx, _ = line.split()
                    self.pcba_scores[(target, int(idx))] = float(score)

        ret = []
        for index in batch.index:
            key = (dataset.target, int(index))
            if key in self.pcba_scores:
                ret.append(self.pcba_scores[key])
            else:
                ret.append(-100)
        return torch.tensor(ret, dtype=torch.float32, device=batch.index.device)

@cache(lambda cfg: "", disable=False)
def get_denvis_pcba_scores(cfg):
    df = pd.read_parquet(cfg.platform.denvis_output_dir + "/litpcba_main_general_surface.parquet")

    df["y_score"] = 0.5*df.y_score_Kd + 0.5*df.y_score_Ki + 0.5*df.y_score_IC50

    pcba_scores = defaultdict(list)
    for full_target, ligand_id, score in zip(tqdm(df.target_id), df.ligand_id, df.y_score):
        target, pdb = full_target.split("#")
        key = (target, int(ligand_id))
        pcba_scores[key].append(score)
    
    pcba_scores = { key: torch.tensor(scores, dtype=torch.float32) for key, scores in pcba_scores.items() }

    return pcba_scores

class DenvisModel(ValModel):

    def __init__(self, cfg):
        self.cfg = cfg
        self.pcba_scores = None
        self.bigbind_scores = {}

    def get_cache_key(self):
        return "denvis"

    def get_name(self):
        return self.get_cache_key()

    def __call__(self, batch, dataset):

        assert isinstance(dataset, LitPcbaDataset)

        if self.pcba_scores is None: 
            self.pcba_scores = get_denvis_pcba_scores(self.cfg)

        ret = []
        for index in batch.index:
            key = (dataset.target, int(index))
            if key in self.pcba_scores:
                ret.append(-torch.median(self.pcba_scores[key]))
            else:
                ret.append(-100)
        return torch.tensor(ret, dtype=torch.float32, device=batch.index.device)


class ComboModel(ValModel):

    def __init__(self, model1: ValModel, model2: ValModel, model1_frac: float):
        self.model1 = model1
        self.model2 = model2
        self.model1_frac = model1_frac
        self.model1_preds = None
        self.model2_preds = None

    def get_cache_key(self):
        return ("combo", self.model1.get_cache_key(), self.model2.get_cache_key(), self.model1_frac)
    
    def get_name(self):
        return f"combo_{self.model1.get_name()}_{self.model2.get_name()}_{self.model1_frac}"

    def init_preds(self, model1_preds, model2_preds):
        self.model1_preds = model1_preds
        self.model2_preds = model2_preds

    def __call__(self, x, dataset):
        raise NotImplementedError()

    def choose_topk(self, k):
        """ returns the indexes of the top k items according to our choice
        hueristic (top model1_frac from model1, top k from those) """
        idx_p1_p2 = list(zip(range(len(self.model1_preds)), self.model1_preds, self.model2_preds))
        random.shuffle(idx_p1_p2)
        k1 = int(len(self.model1_preds)*self.model1_frac)
        top_k1 = sorted(idx_p1_p2, key=lambda x: -x[1])[:k1]
        top_k = sorted(top_k1, key=lambda x: -x[2])[:k]
        return [ idx for idx, p1, p2 in top_k ]
