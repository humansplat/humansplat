from typing import *

import torch
import webdataset as wds
from torch import Tensor


class TarDataset:
    def __init__(self):
        self.dataset_size = None

        self.dataset = None

    def __len__(self):
        assert isinstance(self.dataset_size, int)
        return self.dataset_size

    def get_webdataset(self) -> wds.WebDataset:
        assert isinstance(self.dataset, wds.WebDataset)
        return self.dataset

    ################################ Decode tar bytes ################################

    def decoder(self, sample: Dict[str, Union[str, bytes]]):
        raise NotImplementedError

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        # `sample["__key__"]` has been added to batch samples by `webdataset`, remove it here
        outputs = {
            k: torch.stack([b[k] for b in batch], dim=0)
            for k in batch[0].keys()
            if isinstance(batch[0][k], Tensor)  # only stack tensors
        }
        return outputs
