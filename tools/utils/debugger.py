import sys, pdb, traceback

import torch 
from torch.nn import functional as F

def set_excepthook():
    def excepthook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("\nEntering debugger...")
        pdb.post_mortem(exc_traceback)

    sys.excepthook = excepthook

def detect_signal(attr):
    if attr.startswith("_") or attr == "get_definition_file" or attr == "get_definition_name":
        return False
    else:
        return True

def get_dut_attributes(dut, log, value_rep: str = None):
    log.debug(f"--------------------------------")
    log.debug(f"Getting attributes of {dut}")
    log.debug(f"--------------------------------")
    for attr in dir(dut):
        if detect_signal(attr):
            if value_rep is None:
                try:
                    value = getattr(dut, attr).value
                except:
                    log.debug(f"Cannot get value of {attr}")
            else:
                try:
                    value = getattr(getattr(dut, attr).value, value_rep)
                except:
                    try:
                        value = getattr(dut, attr).value
                    except:
                        log.debug(f"Cannot get value of {attr}")
        else:
            continue
        log.debug(f"{attr}: {value}")

def _get_similarity(tensor_raw, tensor_sim, metric=None):
    if metric == "cosine":
        similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
    elif metric == "pearson":
        similarity = F.cosine_similarity(
            tensor_raw - torch.mean(tensor_raw, dim=-1, keepdim=True),
            tensor_sim - torch.mean(tensor_sim, dim=-1, keepdim=True),
            dim=-1,
        )
    else:
        if metric == "L1_norm":
            similarity = -torch.abs(tensor_raw - tensor_sim)
        elif metric == "L2_norm" or "l2norm" or "l2":
            similarity = -((tensor_raw - tensor_sim) ** 2)
        elif metric == "linear_weighted_L2_norm":
            similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
        elif metric == "square_weighted_L2_norm":
            similarity = -((tensor_raw * (tensor_raw - tensor_sim)) ** 2)
        else:
            raise NotImplementedError(f"metric {metric} not implemented!")
        similarity = torch.mean(similarity, dim=-1)
    return similarity