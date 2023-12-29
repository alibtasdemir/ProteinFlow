"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Dataloader from PDB files.
"""
import logging
import copy
import pickle
import json
import numpy as np
import torch
import tree
from torch.utils import data
import pandas as pd

# from core import utils
# from core import protein
# from core import residue_constants
from utils import residue_constants
from dataset import protein
from utils.pdbUtils import read_pkl, parse_chain_feats


class PdbDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._init_metadata()
        self._diffuser = diffuser

    @property
    def is_training(self):
        return self._is_training

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def diffuser(self):
        return self._diffuser

    def _init_metadata(self):
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv

        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[pdb_csv.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.min_num_chains is not None:
            pdb_csv = pdb_csv[pdb_csv.num_chains >= filter_conf.min_num_chains]
        if filter_conf.max_num_chains is not None:
            pdb_csv = pdb_csv[pdb_csv.num_chains <= filter_conf.max_num_chains]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)
        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        if self._is_training:
            self.csv = pdb_csv
            self._log.info(
                f'Training: {len(self.csv)} examples'
            )
        else:
            all_lengths = np.sort(pdb_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(0.0, 1.0, self._data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len.isin(eval_lengths)]

            # Set a seed
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self._data_conf.samples_per_eval_length, replace=True, random_state=9
            )
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        elif 'chain_name' in csv_row:
            pdb_name = csv_row['chain_name']
        else:
            raise ValueError('Need an identifier.')

        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path)

       #TODO Continue
    def _process_csv_row(self, processed_file_path):
        processed_features = read_pkl(processed_file_path)
        processed_features = parse_chain_feats(processed_features)

        modeled_idx = processed_features['modeled_idx']
        min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)
        del processed_features['modeled_idx']
        processed_features = tree.map_structure(
            lambda x: x[min_idx:(max_idx + 1)], processed_features
        )

        chain_features = {
            'aatype': torch.tensor(processed_features['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_features['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_features['atom_mask']).double()
        }

        #TODO Continue
