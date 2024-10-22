import numpy as np

import selfies as sf
from group_selfies import GroupGrammar

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem, MolToSmiles
from torch import Tensor, long, tensor

from rdkit.DataStructs import TanimotoSimilarity, DiceSimilarity, CosineSimilarity, FingerprintSimilarity

import multiprocessing as mp
from functools import wraps
from itertools import repeat
from typing import Callable, Iterable

NUM_WORKERS = 32

def mp_parallelize(func):
    @wraps(func)
    def wrapper(*args):
        # input validity check
        if not args:
            return []
        lengths = []
        for a in args:
            if isinstance(a, Iterable):
                if len(lengths)>0 and (len(a)!=lengths[-1]):
                    raise ValueError('Input Iterables must be of equal length')
                lengths.append(len(a))
        if len(lengths) == 0:
            raise ValueError('Input arguments must contain at least one Iterable')
        
        args_ = []
        # args may contain list and other non-iterable object
        # the non-iterable objects are broadcast
        for a in args:
            if isinstance(a, Iterable):
                args_.append(a)
            else:
                args_.append(repeat(a, lengths[0]))
        
        # Create a list of argument tuples for starmap
        arg_tuples = list(zip(*args_))
        
        with mp.Pool(processes=NUM_WORKERS) as pool:
            chunksize = max(1, lengths[0] // (NUM_WORKERS * 4))
            results = pool.starmap(func, arg_tuples, chunksize=chunksize)
        
        return results
    return wrapper

def sp_vectorize(func):
    @wraps(func)
    def wrapper(*args):
        # input validity check
        if not args:
            return []
        lengths = []
        for a in args:
            if isinstance(a, Iterable):
                if len(lengths)>0 and (len(a)!=lengths[-1]):
                    raise ValueError('Input Iterables must be of equal length')
                lengths.append(len(a))
        if len(lengths) == 0:
            raise ValueError('Input arguments must contain at least one Iterable')
        
        args_ = []
        # args may contain list and other non-iterable object
        # the non-iterable objects are broadcast
        for a in args:
            if isinstance(a, Iterable):
                args_.append(a)
            else:
                args_.append(repeat(a, lengths[0]))
        
        return list(map(func, *args_))
    
    return wrapper

def is_valid_smiles(smiles: str) -> bool:
    if smiles == '':
        return False
    return MolFromSmiles(smiles, sanitize=True) is not None
    
is_valid_smiles_mp = mp_parallelize(is_valid_smiles)
is_valid_smiles_sp = sp_vectorize(is_valid_smiles)


def canonicalize_smiles(smls: str) -> str:
    '''
    Input must be valid SMILES
    '''
    return Chem.MolToSmiles(Chem.MolFromSmiles(smls), canonical=True)

# Caution: each input must be valid SMILES
canonicalize_smiles_mp = mp_parallelize(canonicalize_smiles)
canonicalize_smiles_sp = sp_vectorize(canonicalize_smiles)


def idxv2str(batch: Tensor, 
             alphabet: list, 
             functional_idx:Iterable[int]=[]) -> str|list[str]:
    '''
    args
        batch: idxv tensor of shape (seq_len, ) or (batch, seq_len), dtype should be torch.long, may contain padding, etc.
        alphabet: the alphabet to decode the indices
        functional_tokens: specify the indices of functional tokens, e.g. [nop] [SOS], these indices are to be dropped

    return
        a list of decoded strings
    '''
    assert batch.dtype is long
    
    omit = set(functional_idx)
    alphabet_ = [ '' if i in omit else a for i, a in enumerate(alphabet)]

    if len(batch.shape) == 1:
        return ''.join([alphabet_[i] for i in batch.tolist()])
    
    elif len(batch.shape) == 2:
        batch = batch.tolist()
        
        results = []
        for t in batch:
            results.append(''.join([alphabet_[i] for i in t]))
        return results
    else:
        raise ValueError(f'Input shape {batch.shape} not supported')


def str2idxv(string: list[str], 
             alphabet: list, 
             tokenizer: Callable,
             default: int=0) -> list[Tensor]:
    '''
    Args
        string: list of strings to deal with
        alphabet: ...
        tokenizer: a function that splits the string into tokens
        default: if a token is not found in alphabet, pad this number

    Return
        a list of tensors of indices
    '''
    atoi = {a: i for i,a in enumerate(alphabet)}
    res = []
    for s in string:
        idxv = [atoi.get(a, default) for a in tokenizer(s)]
        idxv = tensor(idxv, dtype=long)
        res.append(idxv)
    return res


def smiles2selfies(smls: str) -> str:
    '''
    Convert SMILES string to vanilla SELFIES
    
    Returns empty string if input is invalid
    '''
    if is_valid_smiles(smls):
        return sf.encoder(smls)
    else:
        return ''

smiles2selfies_mp = mp_parallelize(smiles2selfies)
smiles2selfies_sp = sp_vectorize(smiles2selfies)


def selfies2smiles(slfs: str) -> str:
    assert isinstance(slfs, str)
    return canonicalize_smiles(sf.decoder(slfs))

selfies2smiles_mp = mp_parallelize(selfies2smiles)
selfies2smiles_sp = sp_vectorize(selfies2smiles)


def smiles2group_selfies(smls: str, grp: GroupGrammar) -> str:
    '''
    Convert SMILES string to Group SELFIES using GroupGrammar grp.

    Returns empty string if input is invalid
    '''
    mol = MolFromSmiles(smls)
    if mol is None:
        return ''
    return grp.full_encoder(mol)


def smiles2group_selfies_sp(smls: list[str], grp: GroupGrammar) -> list[str]:
    '''
    Converts a list of SMILES into Group SELFIES
    '''
    results = []
    for s in smls:
        mol = MolFromSmiles(s)
        if mol is None:
            results.append('')
        results.append(grp.full_encoder(mol))
    return results

smiles2group_selfies_mp = mp_parallelize(smiles2group_selfies)
# def smiles2group_selfies_mp(smls: list[str], grp: GroupGrammar) -> list[str]:
#     '''
#     The multi-processing version of smiles2group_selfies, 
#     use this when there's too many input to handle
#     '''
#     def decode(smls):
#         return smiles2group_selfies(smls, grp)
    
#     with mp.Pool(NUM_WORKERS) as pool:
#         results = pool.map(decode, smls)
#     return results


def group_selfies2smiles(grp_slfs: str, grp: GroupGrammar) -> str:
    '''
    Decode Group SELFIES to canonical SMILES, returns empty string on failure
    '''
    try:
        mol = grp.decoder(grp_slfs)
        return MolToSmiles(mol, canonical=True)
    except:
        return ''
    

def validity_diversity(smls: list[str]) -> tuple[float, float]:
    '''
    Args
        strings: the SMILES strings to deal with

    Return 
        validity ratio, diversity
    '''
    
    is_valid: list[bool] = is_valid_smiles_sp(smls)
    validity = np.mean(np.float64(np.array(is_valid)))
    # calculate diversity
    diversity = len(set(smls))/len(smls)
    return validity, diversity


def tanimoto(original_mol: str|Chem.rdchem.Mol, modified_mol: str|Chem.rdchem.Mol) -> float:
    """
    Calculates tanimoto similarity.

    Args:
        original_smiles: SMILES string or Mol object
        modified_smiles (str): SMILES string or Mol object

    Returns:
        tanimotos similarity (float)
    """
    # Convert SMILES to RDKit molecule objects
    if isinstance(original_mol, str):
        original_mol = Chem.MolFromSmiles(original_mol)
    if isinstance(modified_mol, str):
        modified_mol = Chem.MolFromSmiles(modified_mol)

    # Generate Morgan fingerprints
    fp_original = AllChem.GetMorganFingerprintAsBitVect(original_mol, radius=1, nBits=4096)
    fp_modified = AllChem.GetMorganFingerprintAsBitVect(modified_mol, radius=1, nBits=4096)

    # Calculate similarities
    return TanimotoSimilarity(fp_original, fp_modified)


def calc_similarity(original_smiles: str|Chem.rdchem.Mol, modified_smiles: str|Chem.rdchem.Mol):
    """
    Calculates similarity between two smiles strings.
    GetMorganFingerprintAsBitVect -> turns chemical compound into a feature vector,
                                    and various similarity metrics are applied.

    Args:
        original_smiles (str): SMILES representation of the original molecule.
        modified_smiles (str): SMILES representation of the modified molecule.
        similarity_threshold (float): The lower limit of similarity to stop the optimization.

    Returns:
        dict: Results of various similarity metrics.
    """
    # Convert SMILES to RDKit molecule objects
    if isinstance(original_smiles, str):
        original_mol = Chem.MolFromSmiles(original_smiles)
    if isinstance(modified_smiles, str):
        modified_mol = Chem.MolFromSmiles(modified_smiles)

    # Generate Morgan fingerprints
    fp_original = AllChem.GetMorganFingerprintAsBitVect(original_mol, radius=1, nBits=4096)
    fp_modified = AllChem.GetMorganFingerprintAsBitVect(modified_mol, radius=1, nBits=4096)

    # Calculate similarities
    tanimoto = TanimotoSimilarity(fp_original, fp_modified)
    dice = DiceSimilarity(fp_original, fp_modified)
    cosine = CosineSimilarity(fp_original, fp_modified)
    jaccard = FingerprintSimilarity(fp_original, fp_modified)  # Using Tversky index parameters for Jaccard

    # Convert fingerprints to NumPy arrays for Euclidean distance
    arr_original = np.zeros((2048,), dtype=int)
    arr_modified = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp_original, arr_original)
    DataStructs.ConvertToNumpyArray(fp_modified, arr_modified)
    euclidean = np.linalg.norm(arr_original - arr_modified)

    # Aggregate results
    results = {
        "Tanimoto": tanimoto,
        "Dice": dice,
        "Cosine": cosine,
        "Jaccard": jaccard,
        "Euclidean": euclidean,
    }

    return results

if __name__=='__main__':
    from time import perf_counter

    NUM_WORKERS = 16
    from database import ChemDB

    cdb = ChemDB()
    SMILES = cdb.load_column('qm9', 'SMILES')['SMILES'].to_list()[-5000:]

    start = perf_counter()
    results_single = canonicalize_smiles_sp(SMILES)    
    print(f'Single process: {(perf_counter()-start):.2f}s')
    
    start = perf_counter()
    results_multi = canonicalize_smiles_mp(SMILES)
    print(f'Multi process: {(perf_counter()-start):.2f}s')
    if len(results_single) != len(results_multi):
        print('Results dont match.')

    for i, j in zip(results_single, results_multi, strict=True):
        if not i==j:
            print('Results don\'t match')
            break
    else:
        print('Results all match')


    gg = GroupGrammar.from_file('/home/wang.chenze/ChemVAE/VAE_new/QM9ZINC_BRICS50.txt')
    gselfies = smiles2group_selfies_mp(SMILES, gg)
    print(gselfies[:5])


    print(validity_diversity(SMILES))
    print('Similarity')
    # Example usage
    original_smiles = "O=c1[nH]c(=S)[nH]c2ccccc12"
    modified_smiles = "O=Nc1ccc2c(=O)[nH]c(=S)[nH]c2c1"
    results = calc_similarity(original_smiles, modified_smiles)
    print(results)