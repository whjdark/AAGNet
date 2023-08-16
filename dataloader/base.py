import numpy as np
from tqdm import tqdm
from abc import abstractmethod
import threading

from torch.utils.data import Dataset, DataLoader
import dgl

from utils.data_utils import get_random_rotation, rotate_uvgrid
from utils.data_utils import load_json_or_pkl, load_statistics
from utils.data_utils import standardization, center_and_scale


class BaseDataset(Dataset):
    @staticmethod
    @abstractmethod
    def num_classes():
        pass

    def __init__(self, transform, random_rotate) -> None:
        super().__init__()
        self.transform = transform
        self.random_rotate = random_rotate
        self.dataset = None

    def graphs(self):
        return self.dataset

    def process_chunk(self, 
                      chunk, 
                      split_file_list, 
                      normalization_attribute, 
                      center_and_scale_grid, 
                      stat):
        result = []
        for one_data in chunk:
            fn, data = one_data
            if fn in split_file_list:
                one_graph = self.load_one_graph(fn, data)
                if one_graph is None:
                    continue
                if one_graph["graph"].edata["x"].size(0) == 0:
                    # Catch the case of graphs with no edges
                    continue
                if normalization_attribute:
                    one_graph = standardization(one_graph, stat)
                if center_and_scale_grid:
                    one_graph = center_and_scale(one_graph)
                result.append(one_graph)
        return result

    def load_graphs(self, 
                    file_path, 
                    graphs=None, 
                    split_file_list=None, 
                    center_and_scale_grid=True, 
                    normalization_attribute=True, 
                    num_threads=4):
        self.data = []
        if graphs:
            self.dataset = graphs
        else:
            self.dataset = load_json_or_pkl(file_path.joinpath('graphs.json'))
        if normalization_attribute:
            stat = load_statistics(file_path.joinpath('attr_stat.json'))

        # divide the dataset into num_threads chunks
        chunk_size = (len(self.dataset) + num_threads - 1) // num_threads
        chunks = [self.dataset[i:i+chunk_size] for i in range(0, len(self.dataset), chunk_size)]

        # create threads and process each chunk
        threads = []
        results = [[] for _ in range(num_threads)]
        for i in range(num_threads):
            t = threading.Thread(target=lambda i: results[i].extend(
                self.process_chunk(
                    chunks[i], split_file_list, 
                    normalization_attribute, center_and_scale_grid, stat)), args=(i,))
            threads.append(t)
            t.start()

        # wait for all threads to finish
        for t in threads:
            t.join()

        # combine results from all threads
        self.data = [item for sublist in results for item in sublist]

    def load_one_graph(self, fn, data):
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # use data augmentation
        data = self.data[idx]
        if self.random_rotate:
            rotation = get_random_rotation()
            data["graph"].ndata["grid"] = rotate_uvgrid(data["graph"].ndata["grid"], rotation)
            if "grid" in data["graph"].edata.keys():
                data["graph"].edata["grid"] = rotate_uvgrid(data["graph"].edata["grid"], rotation)
        if self.transform:
            data["graph"] = self.transform(data["graph"])
        return data

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames}

    def get_dataloader(self, batch_size=128, shuffle=True, sampler=None, num_workers=0, drop_last=True, pin_memory=False):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=drop_last,
            pin_memory=pin_memory
        )