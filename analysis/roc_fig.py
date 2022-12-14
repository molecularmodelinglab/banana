
import matplotlib.pyplot as plt
import torch

from common.cfg_utils import get_config
from validation.validate import validate

def plot_many_rocs(ax, rocs, title):
    for name, roc in rocs.items():
        fpr, tpr, thresh = roc
        ax.plot(fpr.cpu(), tpr.cpu(), label=name)
    ax.plot([0, 1], [0, 1], color='black')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    # ax.legend()

def fake_validate(*args, **kwargs):
    return {
        "roc": (
            torch.tensor([0.0, 0.5, 1.0]),
            torch.tensor([0.0, 0.75, 1.0]),
            None
        )
    }

def make_roc_figs(cfg, tag, split):
    fig, axs = plt.subplots(2, 2)

    run_ids = {
        "Ligand and receptor": "37jstv82",
        "Ligand only": "exp293if",
    }
    for test_sna in [ False, True]:
        rocs = {}
        for name, run_id in run_ids.items():
            print(f"Validating SNA {name}")
            rocs[name] = fake_validate(cfg, run_id, tag, split, sna_override=test_sna)["roc"]
        plot_many_rocs(axs[0][int(test_sna)], rocs, f"Train SNA, Test SNA {test_sna}")

        run_ids = {
            "Ligand and receptor": "1es4be17",
            "Ligand only": "1qwd5qn6",
        }
        rocs = {}
        for name, run_id in run_ids.items():
            print(f"Validating Non-SNA {name}")
            rocs[name] = fake_validate(cfg, run_id, tag, split, override=test_sna)["roc"]
        plot_many_rocs(axs[1][int(test_sna)], rocs, f"Train without SNA, Test SNA {test_sna}")

    fig.tight_layout()
    fig.set_size_inches(6, 6)
    fig.suptitle("With SNA")
    fig.legend(["Ligand and Receptor", "Ligand Only"])
    fig.savefig("./outputs/roc.pdf", dpi=300)

if __name__ == "__main__":
    cfg = get_config()
    make_roc_figs(cfg, "v4", "test")