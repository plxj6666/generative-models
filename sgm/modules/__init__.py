from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "svd.sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
