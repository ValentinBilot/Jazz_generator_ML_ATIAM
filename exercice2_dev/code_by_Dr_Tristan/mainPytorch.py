#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:43:07 2018

@author: carsault
"""

# %%
import torch
from random import randint
from utilities import chordUtil
from utilities import dataImport
from utilities.chordUtil import *
from utilities.dataImport import *
from sklearn.model_selection import train_test_split
import os

# %%
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print("use_cuda")
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

# Init
lenSeq = 16
alpha = 'a0'
rootname = "inputs/jazz_xlab/"
filenames = os.listdir(rootname)
# filenames.remove(".DS_Store")
dictChord, listChord = chordUtil.getDictChord(eval(alpha))

# Create datasets
files_train, files_test = train_test_split(filenames, test_size=0.7)
dataset_train = dataImport.ChordSeqDataset(
    files_train, rootname, alpha, dictChord, lenSeq)
dataset_test = dataImport.ChordSeqDataset(
    files_test, rootname, alpha, dictChord, lenSeq)

# Create generators
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
training_generator = data.DataLoader(dataset_train, **params)
testing_generator = data.DataLoader(dataset_test, **params)


print_every = 10
# Begin training
max_epochs = 500
for epoch in range(max_epochs):
    # Training
    if epoch % print_every == 0:
        print(epoch)
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(
            device), local_labels.to(device)

        # //// Train the model  ////

    # Testing
    for local_batch, local_labels in testing_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(
            device), local_labels.to(device)

        # //// Test the model  ////
