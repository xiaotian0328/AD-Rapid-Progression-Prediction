import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    Dataset, 
    DataLoader
)

def set_binary_labels(labels):
    threshold_10 = labels.quantile(0.1)
    threshold_90 = labels.quantile(0.9)
    
    # adas > q90
    y_adas_q90 = (labels['ADAS_ADASTS14'] > threshold_90['ADAS_ADASTS14']).astype(int).to_numpy()

    # adl < q10
    y_adl_q10 = (labels['ADL_ADLOVRALS'] < threshold_10['ADL_ADLOVRALS']).astype(int).to_numpy()
    
    # cdr > q90
    y_cdr_q90 = (labels['CDR_CDRTS'] > threshold_90['CDR_CDRTS']).astype(int).to_numpy()
    
    # mmse < q10
    y_mmse_q10 = (labels['MMSE_MMSETS'] < threshold_10['MMSE_MMSETS']).astype(int).to_numpy()
    
    return {
        "adas_q90": y_adas_q90,
        "adl_q10": y_adl_q10,
        "cdr_q90": y_cdr_q90,
        "mmse_q10": y_mmse_q10,
        "adas_adl": np.logical_and(y_adas_q90, y_adl_q10).astype(int),
        "adas_cdr": np.logical_and(y_adas_q90, y_cdr_q90).astype(int),
        "adas_mmse": np.logical_and(y_adas_q90, y_mmse_q10).astype(int),
        "adl_cdr": np.logical_and(y_adl_q10, y_cdr_q90).astype(int),
        "adl_mmse": np.logical_and(y_adl_q10, y_mmse_q10).astype(int),
        "cdr_mmse": np.logical_and(y_cdr_q90, y_mmse_q10).astype(int),
    }


class ADDataset(Dataset):
    def __init__(
        self, 
        X_static, 
        X_dynamic, 
        Y, 
        X_dynamic_valid_lens=None
    ):
        super(ADDataset, self).__init__()
        self.X_static = X_static
        self.X_dynamic = X_dynamic
        self.X_dynamic_valid_lens = X_dynamic_valid_lens
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        if self.X_dynamic_valid_lens is None:
            return (
                self.X_static[idx].astype(np.float32), 
                self.X_dynamic[idx].astype(np.float32), 
                self.Y[idx].astype(np.float32)
            )
        else:
            return (
                self.X_static[idx].astype(np.float32), 
                self.X_dynamic[idx].astype(np.float32), 
                self.X_dynamic_valid_lens[idx].astype(np.float32), 
                self.Y[idx].astype(np.float32)
            )
    
    
class BaselineDataset(Dataset):
    def __init__(
        self, 
        X_static, 
        Y
    ):
        super(BaselineDataset, self).__init__()
        self.X_static = X_static
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_static[idx].astype(np.float32),
            self.Y[idx].astype(np.float32),
        )
    