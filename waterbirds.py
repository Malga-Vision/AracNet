#!/usr/bin/env python3

# https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
from pathlib import Path
import numpy as np
import torch
import os
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import List, Callable, Tuple, Generator, Union
from collections import OrderedDict
from torch.utils.data import ConcatDataset
import pandas as pd
import requests
from tqdm import tqdm
import tarfile
    
#os.chdir('./Aracnonet/')

data_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class Waterbirds(Dataset):
    DOWNLOAD_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
    DATASET_NAME = "waterbirds"

    def __init__(self, env: str, root: str = "./data", transform = data_transform, metadata_filename: str = "metadata.csv", return_index: bool = True):
        self.root:              str  = os.path.join(root, Waterbirds.DATASET_NAME)
        self.env:               str  = env
        self.metadata_filename: str  = metadata_filename
        self.return_index:      bool = return_index
        self.num_classes = 2

        self.env_to_split = {
            "train": 0,
            "val":   1,
            "test":  2
        }

        if "waterbirds" not in os.listdir("./data"):
            self.__download_dataset()
        else: self.root = "./data/waterbirds/waterbird_complete95_forest2water2"

        self.transform = transform
        self.metadata_path = os.path.join(self.root, self.metadata_filename)

        metadata_csv = pd.read_csv(self.metadata_path)
        metadata_csv = metadata_csv.query(f"split == {self.env_to_split[self.env]}")

        self.samples = {}
        self.files_count = 0
        for i, (_, sample_info) in enumerate(metadata_csv.iterrows()):
            self.samples[i] = {
                "img_id":    sample_info["img_id"],
                "image_path":  os.path.join(self.root, sample_info["img_filename"]),
                "class_label": int(sample_info["y"]),
                "bias_label":  int(sample_info["place"]),
                "all_attrs": list((str(e) for e in sample_info))
            }
            self.files_count += 1

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "waterbirds.tar.gz")
        print(f"=> Downloading {Waterbirds.DATASET_NAME} for {Waterbirds.DOWNLOAD_URL}")

        try:
            response = requests.get(Waterbirds.DOWNLOAD_URL, stream=True)
            response.raise_for_status()

            with open(output_path, mode="wb") as write_stream, tqdm(
                desc=output_path,
                total=int(response.headers["content-length"], 0),
                unit="B",
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    write_stream.write(chunk)
                    pbar.update(len(chunk))

        except:
            raise RuntimeError("Unable to complete dataset download, check for your internet connection or try changing download link.")
        
        print(f"=> Extracting waterbird_complete95_forest2water2.tar.gz to directory {self.root}")
        try:
            with tarfile.open(output_path, mode="r:gz") as unballer:
                unballer.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

        self.root = os.path.join(self.root, "waterbird_complete95_forest2water2")
        os.remove(output_path)
        


    def __len__(self) -> int:
        return self.files_count

    def __getitem__(self, index: Union[int, slice, list]) -> Tuple[torch.Tensor]:
        if isinstance(index, slice):
            return [self.__getitem__(i) for i in range(*index.indices(len(self)))]
        
        if isinstance(index, list):
            return [self.__getitem__(idx) for idx in index]

        image = self.transform(Image.open(self.samples[index]["image_path"]))
        class_label = self.samples[index]["class_label"]
        bias_label  = self.samples[index]["bias_label"]
        img_id = self.samples[index]["img_id"]

        return image, (class_label, bias_label), index
    
    def perclass_populations(self, return_labels: bool = False) -> Union[Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1][0]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts
    
    def get_bias_labels(self) -> Generator[None, None, torch.Tensor]:
        for i in range(len(self)):
            yield self[i][2]

    def __repr__(self) -> str:
        return f"Waterbirds(env={self.env}, bias_amount=Fixed, num_classes={self.num_classes})"
    

if __name__ == "__main__":
    
    Waterbirds(env="train")