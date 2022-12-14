
import matplotlib.pyplot as plt
import torch

from common.cfg_utils import get_config
from validation.validate import validate

def plot_many_rocs(ax, rocs, aucs, title):
    for name, roc in rocs.items():
        fpr, tpr, thresh = roc
        ax.plot(fpr.cpu(), tpr.cpu(), label=f"{name} (AUC {aucs[name]:.2f})")
    ax.plot([0, 1], [0, 1], color='black')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_title(title)
    ax.legend()

def fake_validate(*args, **kwargs):
    return {
        "roc": (
            torch.tensor([0.0, 0.5, 1.0]),
            torch.tensor([0.0, 0.75, 1.0]),
            None
        ),
        "auroc": 0.7
    }

def make_roc_figs(cfg, tag, split):
    fig, axs = plt.subplots(2, 2)

    for test_sna in [ False, True]:

        run_ids = {
            "Full model": "1es4be17",
            "Ligand only": "1qwd5qn6",
        }
        rocs = {}
        aucs = {}
        for name, run_id in run_ids.items():
            print(f"Validating Non-SNA {name}")
            metrics = validate(cfg, run_id, tag, split, sna_override=test_sna)
            rocs[name] = metrics["roc"]
            aucs[name] = metrics["auroc"]
        plot_many_rocs(axs[0][int(test_sna)], rocs, aucs, f"Train without SNA, Test SNA {test_sna}")

        run_ids = {
            "Full model": "37jstv82",
            "Ligand only": "exp293if",
        }
        rocs = {}
        aucs = {}
        for name, run_id in run_ids.items():
            print(f"Validating SNA {name}")
            metrics = validate(cfg, run_id, tag, split, sna_override=test_sna)
            rocs[name] = metrics["roc"]
            aucs[name] = metrics["auroc"]
        plot_many_rocs(axs[1][int(test_sna)], rocs, aucs, f"Train SNA, Test SNA {test_sna}")

    pad = 5 # in points

    rows = [ "Train without SNA", "Train with SNA"]
    cols = [ "Test without SNA", "Test with SNA"]

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation='vertical')

    fig.tight_layout()
    fig.subplots_adjust(left=0.15)
    fig.set_size_inches(8, 8)
    # fig.legend(["Ligand and Receptor", "Ligand Only"])
    fig.savefig("./outputs/roc.pdf", dpi=300)

if __name__ == "__main__":
    cfg = get_config()
    make_roc_figs(cfg, "v4", "test")