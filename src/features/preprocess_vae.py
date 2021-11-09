#!/usr/bin/env pythonf
"""preprocess_vae.py: script that contains the preprocessing methods prior to loading into pytorch """

import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import torch


def metadata_mapping(encodings_file, metadata_path):
    """Reads metadata and list of encodings and outputs list of targets
    Parameters
    ----------
    encodings_file : path to encoding file
    metadata_path   : path to metadata file

    Returns
    -------
    targets         : list of targets linked to X
    """    
    # Get list of encodings
    with open (encodings_file, 'rb') as fp:
        encodings = pickle.load(fp)

    # Open metadata file
    metadata = pd.read_csv(metadata_path, sep="\t", names=["index", "index2", "encoding", "encoding2", "time_num", "time_str", "Country", "SNP_cov", "SNP_cov2", "SNP_cov3", "Annotated_ancestry"], header=None)

    # Replace nans by "Unknown"
    metadata = metadata.replace(np.nan, "Unknown")

    # Get patient encoded IDs and map to targets
    targets = list()
    targets = metadata[metadata["encoding"].isin(encodings)].Annotated_ancestry.to_list()

    print("--> Metadata mapped for all data")
    return targets


def one_hot_encoding(list_to_encode):
    """Hot encodes a list of strings
    Parameters
    ----------
    list_to_encode : list containing strings

    Returns
    -------
    one_hot_encoded : one-hot encoded numpy array
    """
    ### One hot encoding
    # Implement your code
    return encoded_array


def split_train_test(X, targets, prop):
    """Splits X and targets np.arrays into train and test according to prop
    Parameters
    ----------
    X : vector of inputs
    targets : vector of targets
    prop : floating point corresponding to the training test partition
    
    Returns
    -------
    X_train : X * prop
    X_test X * (1-prop)
    targets_train : targets * prop
    targets_test : targets * (1-prop)

    X_train = X[:int(len(X)*prop)]
    X_test = X[int(len(X)*prop+1):]
    
    y_train = targets[:int(len(X)*prop)]
    y_test = targets[int(len(X)*prop+1):]
    """
    # Implement your code
    return X_train, X_test, y_train, y_test


def impute_data(tensor, frequency_df="no_frequency", batch_size=20, impute_all=False, categorical=False):
    """Replaces tensor values with nas with zeroes
    Parameters
    ----------
    X : tensor 
    
    variants : path to list of variants matching the tensor 
    
    Returns
    -------
    tensor with replaced values
    """
    index_nan = (tensor!=tensor).nonzero()

    if categorical is True:
        tensor = tensor.cpu()
        shape = tensor.shape

        # Get index X == nan and frequencies for those
        tensor =  np.array(tensor)
        index_nan = np.where(np.isnan(tensor))
        tensor[index_nan] = np.zeros_like(tensor[index_nan])
        tensor = torch.FloatTensor(tensor)

        assert len((tensor!=tensor).nonzero()) == 0
        return tensor
            

def get_enc_dict(original_targets, targets):
    """Takes a list of encoded targets and original strings and makes a dictionary of the encoding
    Parameters
    ----------
    original_targets : str targets
    targets : encoded targets
    
    Returns
    -------
    dictionary with encoding mapping
    """
    targets = tuple(map(tuple, targets))

    # Initialize df
    df = pd.DataFrame({"targets":list(targets), "original_targets":original_targets})
    df = df.drop_duplicates(subset=["original_targets"])

    # Create dict 
    dict_encoding = defaultdict()
    dict_encoding = pd.Series(df.original_targets.values, index=df.targets.values).to_dict()

    return dict_encoding


def loss_ignore_nans(loss, x):
    """Takes loss values and multiplies losss by zero for values corresponding to a np.nan"""
    # Implement your code
    return loss




