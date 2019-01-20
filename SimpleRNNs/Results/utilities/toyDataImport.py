#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:45:20 2018

@author: carsault, adapted by valentin Bilot
"""

# %%
import torch
from random import randint
from torch.utils import data
from utilities import chordUtil


class ChordSeqDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, root, alpha, dictChord, lenSeq):
        'Initialization'
        self.list_IDs = list_IDs
        self.root = root
        self.alpha = alpha
        self.dictChord = dictChord
        self.lenSeq = lenSeq

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Open xlab (not sure if it is really xlab format tho)
        xlab = open(self.root + ID, "r")
        lines = xlab.read().split("\n")
        # Transform with one chord by beat
        chordBeat = []
        # Initialize with N
        for i in range(self.lenSeq):
            chordBeat.append(
                self.dictChord[chordUtil.reduChord('N', self.alpha)])
        # Complete with chords in the file
        for i in range(1, len(lines)):
            line = lines[i-1].split(" ")
            chordBeat.append(
                self.dictChord[chordUtil.reduChord(line[1], self.alpha)])
        # Select a random section in the file
        X = torch.zeros(self.lenSeq, len(self.dictChord))
        start = randint(0, len(chordBeat)-1-self.lenSeq)
        for i in range(self.lenSeq):
            X[i][chordBeat[start+i]] = 1
        # Get label
        y = torch.zeros(len(self.dictChord))
        y[chordBeat[start+self.lenSeq]] = 1
        #y = torch.zeros(1)
        #y[0] = chordBeat[start+self.lenSeq]

        return X, y