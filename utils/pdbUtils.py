import io
import pickle
from typing import Optional, Any
from Bio.PDB import PDBParser
import torch

from dataset.protein import Protein
import residue_constants
import numpy as np
import os


class CpuUnpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.
    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda x: torch.load(io.BytesIO(x), map_location='cpu')
        else:
            return super().find_class(module, name)


def write_pkl(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch: bool = False):
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, "wb") as f:
            pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        try:
            with open(read_path, "rb") as f:
                return CpuUnpickler(f).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\nSecond error: {e2}')
            raise e


def build_from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

  WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

  Args:
    pdb_str: The contents of the pdb file
    chain_id: If chain_id is specified (e.g. A), then only that chain
      is parsed. Otherwise all chains are parsed.

  Returns:
    A new `Protein` parsed from the pdb contents.
  """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB contains an insertion code at chain {chain.id} and residue '
                    f'index {res.id[1]}. These are not supported.')
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors))
