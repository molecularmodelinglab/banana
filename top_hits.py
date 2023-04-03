import pandas as pd
import sys
from tqdm import trange

pocket_file = sys.argv[1]
prefix = pocket_file.split(".")[0]
out_filename = f"/proj/kpoplab/banana_outputs/{prefix}_scores.csv"

out_csv = open(out_filename, "w")
out_csv.write("smiles,idnumber,score\n")
scores = []
for i in trange(800):
    out_name = f"screen_outputs_{pocket_file}_{i}.txt"
    smiles_name = f"/proj/kpoplab/Enamine/44M_diversity_set.csv_{str(i+1).zfill(3)}"
    with open(out_name, "r") as f:
        with open(smiles_name, "r") as smi:
            smi = iter(smi)
            if i == 0:
                next(smi)
            for line, smi_line in zip(f, smi):
                compound, idnum = smi_line.strip().split(",")
                score, valid = line.split(",")
                if valid:
                    scores.append(score)
                    out_csv.write(f"{compound},{idnum},{score}\n")
out_csv.close()

df = pd.read_csv(out_filename)
df = df.query("score > 0.0")
df = df.sort_values(by="score", ascending=False)
df.to_csv(f"/proj/kpoplab/banana_outputs/{prefix}_hits.csv", index=False)
df.head(1000000).to_csv(f"/proj/kpoplab/banana_outputs/{prefix}_1M_top_hits.csv", index=False)
