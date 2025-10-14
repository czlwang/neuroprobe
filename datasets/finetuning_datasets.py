import random
import os
import torch
from tqdm import tqdm as tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils import data
from datasets import register_dataset
from preprocessors import build_preprocessor
from pathlib import Path
import logging

log = logging.getLogger(__name__)

class BaseFinetuning(data.Dataset):
    def __init__(self, preprocessor_cfg=None):
        super().__init__()

        self.extracter = build_preprocessor(preprocessor_cfg)

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def get_output_size(self):
        return 1 #single logit

    def __len__(self):
        return self.X_data.shape[0]

    def label2idx(self, label):
        return self.label2idx_dict[label]

@register_dataset(name="finetuning")
class FinetuningDataset(BaseFinetuning):
    #NOTE: this is to be used in pre-training, while the other class in this file is to be used during fine-tuning
    def __init__(self, X_data, y_data, preprocessor_cfg=None) -> None:
        super().__init__(preprocessor_cfg=preprocessor_cfg)
        self.X_data = X_data
        self.y_data = y_data
        
        self.label2idx_dict = {x:y for (x,y) in enumerate(sorted(list(set(y_data))))}
        self.idx2label_dict = {y:x for (x,y) in self.label2idx_dict.items()}

    def __getitem__(self, idx: int):

        #NOTE: remember not to load to cuda here
        wav = self.X_data[idx][np.newaxis,:].astype('float32')
        specs = self.extracter(wav)
        length = specs.shape[0]
        return {
                "input" : specs,
                "length": length,
                "wav": wav,
                "label": self.y_data[idx]
               }
