import torch
import os, glob

from utils.pdbUtils import read_pkl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Import data
    # train_data = ProteinDataset()


if __name__ == '__main__':
    FEATURES_1D = (
        "coords_in",
        "torsions_in",
        "b_factors",
        "atom_positions",
        "aatype",
        "atom_mask",
        "residue_index",
        "chain_index",
    )
    data_dir = "data/processed/"
    prot = read_pkl(os.path.join(data_dir, "1A0N.pkl"))
    print(prot.keys())
    # print(prot["atom_positions"].shape)
