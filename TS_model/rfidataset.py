import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class NenuFARDatasetH5(Dataset):
    def __init__(self, h5_path, transform=None, return_meta=False):
        self.h5_path = h5_path
        self.transform = transform
        self.return_meta = return_meta

        self.f = h5py.File(self.h5_path, "r")
        self.X = self.f["X"]              
        self.Y = self.f["Y"]              
        self.orig_idx = self.f["orig_index"][:]   
        self.labels   = self.f["label"][:]        
        self.frac     = self.f["frac_flagged"][:] 

        self.N = self.X.shape[0]
        print("Nb samples:", self.N)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        
        img = self.X[idx]     
        mask = self.Y[idx]    


        image = torch.from_numpy(img).unsqueeze(0).float()   
        mask_t = torch.from_numpy(mask).long()               

        if not self.return_meta:
            return image, mask_t

        meta = {
            "orig_index": int(self.orig_idx[idx]),
            "label": int(self.labels[idx]),
            "frac_flagged": float(self.frac[idx]),
        }
        return image, mask_t, meta