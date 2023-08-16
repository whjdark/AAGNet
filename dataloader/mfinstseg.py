import pathlib
import json
import os
import math

import torch
import dgl
import numpy as np

from .base import BaseDataset
from utils.data_utils import load_one_graph



class MFInstSegDataset(BaseDataset):
    remap_dict = {1:0, 12:1, 14:2, 6:3, 0:4, 23:5, 24:6}
    
    @staticmethod
    def num_classes(type='full'):
        return 25 if type == 'full' else 7
    
    @staticmethod
    def remap(logit):
        '''
        if you choose tiny dataset, you should remap the semantic labels
        from full dataset logits (0-25) to tiny dataset logits (0-7)
        Through hole: 1 -> 0, 
        Blind hole: 12 -> 1, 
        Rectangular pocket: 14 -> 2, 
        Rectangular through slot: 6 -> 3, 
        Round: 0 -> 4, 
        Chamfer: 23 -> 5,
        stock: : 24 -> 6
        '''
        return MFInstSegDataset.remap_dict[logit]
    
    def __init__(self, 
                 root_dir, 
                 graphs=None, 
                 split="train", 
                 normalize=True, 
                 center_and_scale=True, 
                 random_rotate=False, 
                 num_train_data=-1, 
                 transform=None,
                 dataset_type='full',
                 num_threads=0):
        """
        Load the MFInstSeg Dataset from the root directory.

        Args:
            root_dir (str): Root path of the dataset.
            graphs (list, optional): List of graph data.
            split (str, optional): Data split to load. Defaults to "train".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
            num_train_data (int, optional): Number of training examples to use. Defaults to -1 (all training examples will be used).
            transform (callable, optional): Transformation to apply to the data.
            dataset_type (str, optional): Type of dataset to load. Defaults to "full".
            num_threads (int, optional): Number of threads to use for data loading. Defaults to 0.
        """
        path = pathlib.Path(root_dir)
        self.path = path
        self.transform = transform
        self.random_rotate = random_rotate
        self.dataset_type = dataset_type # full or tiny
        assert split in ("train", "val", "test", "all")

        # filelist = self._get_filenames(root_dir)
        # # 70% for train, 15% for valid, 15% for test
        # train_split = int(0.7 * len(filelist))
        # valid_split = int(0.15 * len(filelist))
        # if split == "train":
        #     split_filelist = filelist[:train_split][:num_train_data]
        # elif split == "val":
        #     split_filelist = filelist[train_split:train_split+valid_split]
        # elif split == "test":
        #     split_filelist = filelist[train_split+valid_split:]
        # elif split == "all":
        #     split_filelist = filelist
        
        # load data partition from train.txt valid.txt test.txt file
        # load data partition from train.txt valid.txt test.txt file
        if split == "all":
            with open(os.path.join('MFInstseg_partition', 'train.txt'), 'r') as f:
                train_filelist = f.readlines()
                train_filelist = [x.strip() for x in train_filelist]
            with open(os.path.join('MFInstseg_partition', 'val.txt'), 'r') as f:
                valid_filelist = f.readlines()
                valid_filelist = [x.strip() for x in valid_filelist]
            with open(os.path.join('MFInstseg_partition', 'test.txt'), 'r') as f:
                test_filelist = f.readlines()
                test_filelist = [x.strip() for x in test_filelist]
            assert len(train_filelist) != 0 and len(valid_filelist) != 0 and len(test_filelist) != 0, \
                'have empty partition file'
            split_filelist = train_filelist + valid_filelist + test_filelist
        else:
            with open(os.path.join('MFInstseg_partition', split+'.txt'), 'r') as f:
                split_filelist = f.readlines()

        if split == "train":
            split_filelist = split_filelist[:num_train_data]
            
        assert len(split_filelist) != 0, 'have empty partition file'
        split_filelist = [x.strip() for x in split_filelist]
        # Load graphs
        print(f"Loading {split} data...")
        split_filelist = set(split_filelist)
        graph_path = path.joinpath("aag")
        self.load_graphs(graph_path, graphs, split_filelist, center_and_scale, normalize, num_threads)
        print("Done loading {} files".format(len(self.data)))

    def _get_filenames(self, root_dir):
        """
        Get a list of filenames for the data split specified in the `split` argument.

        Args:
            root_dir (str): Root path of the dataset.

        Returns:
            List[str]: List of filenames.
        """
        step_dir = os.path.join(root_dir, 'steps')
        step_dir = pathlib.Path(step_dir)
        files = list(
            x.stem for x in step_dir.rglob(f"*.st*p")
        )
        return files

    def _collate(self, batch):
        """
        Collate a batch of data samples together into a single batch.

        Args:
            batch (List[dict]): List of data samples.

        Returns:
            dict: Batched data.
        """
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        inst_labels = self.pack_pad_2D_adj(batch)
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, 
                "inst_labels": inst_labels, 
                "filename": batched_filenames}
    
    def pack_pad_2D_adj(self, batch):
        """
        Pack and pad the 2D adjacency matrix for each graph in the batch.
        """
        max_num_nodes = max([sample["inst_y"].shape[0] for sample in batch])
        batched_adj = torch.zeros(len(batch), max_num_nodes, max_num_nodes, dtype=torch.float)
        for i, sample in enumerate(batch):
            adj = sample["inst_y"]
            num_nodes = sample["inst_y"].shape[0]
            batched_adj[i, :num_nodes, :num_nodes] = adj
        return batched_adj

    def load_one_graph(self, fn, data):
        """
        Load the data for a single file.

        Args:
            fn (str): Filename.
            data (dict): Data for the file.

        Returns:
            dict: Data for the file.
        """
        # Load the graph using base class method
        sample = load_one_graph(fn, data)
        num_faces = sample['graph'].num_nodes() 
        # Additionally load the label and store it as node data
        label_file = self.path.joinpath("labels").joinpath(fn + ".json")
        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)
        _, labels = labels_data[0]
        seg_label, inst_label, bottom_label = labels['seg'], labels['inst'], labels['bottom']
        assert len(seg_label) == len(inst_label) and len(seg_label) == len(bottom_label), \
            'have wrong label: '+ fn
        assert num_faces == len(seg_label), \
            'File {} have wrong number of labels {} with AAG faces {}. '.format(
                fn, len(seg_label), num_faces)
        # read semantic segmentation label for each face
        face_segmentaion_labels = np.zeros(num_faces)
        for idx, face_id in enumerate(range(num_faces)):
            index = seg_label[str(face_id)]
            face_segmentaion_labels[idx] = self.remap(index) if self.dataset_type == 'tiny' else index
        # read instance segmentation labels for each instance
        # just a face adjacency
        instance_label = np.array(inst_label, dtype=np.int32)
        # read bottom face segmentation label for each face
        bottom_segmentaion_labels = np.zeros(num_faces)
        for idx, face_id in enumerate(range(num_faces)):
            index = bottom_label[str(face_id)]
            bottom_segmentaion_labels[idx] = index
        # to torch array
        sample["graph"].ndata["seg_y"] = torch.tensor(face_segmentaion_labels).long()
        sample["inst_y"] = torch.tensor(instance_label).float()
        sample["graph"].ndata["bottom_y"] = torch.tensor(bottom_segmentaion_labels).float().reshape(-1,1)
        return sample


if __name__ == '__main__':
    dataset = MFInstSegDataset(root_dir='E:\\AAGNet\\dataset\\data', split='test', center_and_scale=True, normalize=False)
    print(dataset[0])
    