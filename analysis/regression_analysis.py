from common.cfg_utils import get_config
from validation.validate import validate

def validate_regression(cfg):
    print("Validating Ligand and Receptor on test set")
    validate(cfg, "1jy15xne", "latest", "test")
    print("Validating Ligand only on test set")
    validate(cfg, "49nua94a", "latest", "test")


    print("Validating Ligand and Receptor on train set")
    validate(cfg, "1jy15xne", "latest", "train")
    print("Validating Ligand only on train set")
    validate(cfg, "49nua94a", "latest", "train")

if __name__ == "__main__":
    cfg = get_config()
    validate_regression(cfg)