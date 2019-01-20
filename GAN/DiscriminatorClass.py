import torch
from random import randint
from utilities import chordUtil
from utilities import dataImport
from utilities import plotAndTimeUtil
from utilities.chordUtil import *
from utilities.dataImport import *
from utilities.plotAndTimeUtil import *
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
import sys
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utilities import chordsDistances
from utilities.chordsDistances import getPaulMatrix
from utilities import remapChordsToBase
from utilities.remapChordsToBase import remapPaulToTristan
from random import choices


class MYDis(nn.Module):

    # Defining Pytorch things to make the Network

    def __init__(self, input_size, hidden_size_discriminator, num_layers_discriminator, dropout_discriminator):
        super(MYDis, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size_discriminator,
                          num_layers=num_layers_discriminator, batch_first=True, dropout=dropout_discriminator)
        self.last_fully_connected = nn.Linear(hidden_size_discriminator, 1)
        # not sure about the dim = 1 in softmax
        #self.softmax = nn.LogSoftmax(dim=1)

        # Usefull for monitoring
        #self.trainingData = [[0],["None"],["None"],[[0]],[[0]],[[0]],[[0]],[0],[0],["No"]]

    def forward(self, input_batch):
        output, hidden = self.rnn(input_batch)
        output = output[:, -1, :]
        #print("outputsize : ")
        # print(output.size())
        output = self.last_fully_connected(output)
        #print("outputsize : ")
        # print(output.size())
        # not sure about the dim = 1 in softmax
        #output = self.softmax(output)
        #print("outputsize : ")
        # print(output.size())
        return output

    def trainOnGen(self, model_type, print_every, optimizer, lossFunction, genNet, alphabet='a0', sequence_lenght_in=16, sequence_lenght_out=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size_discriminator=128, num_layers_discriminator=2, dropout_discriminator=0.1, learning_rate_discriminator=1e-4, epochs=10, sampling=True):
        self.train(mode=True)
        genNet.train(mode=False)
        if using_cuda:
            print("Trying using Cuda ...")
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                print("OK")
            else:
                print("Woops, Cuda cannot be found :'( ")
            device = torch.device("cuda:0" if use_cuda else "cpu")
        else:
            device = torch.device("cpu")
            print("Using Cpu")
        self.to(device)

        start = time.time()

        all_losses, similarity = self.doEpochsOnGen(model_type, epochs, print_every, optimizer, lossFunction, learning_rate_discriminator,
                                                    batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, genNet, sampling, training=True)

        print("Finished Training")
        return similarity

    def doEpochsOnGen(self, model_type, epochs, print_every, optimizer, lossFunction, learning_rate_discriminator, batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, genNet, sampling, training=True):

        # Init Training results and monitoring data
        all_losses = []
        total_loss = 0  # Reset every plot_every iters
        disOnGenWin = 0
        totalSize = 0
        disOnGenWinTest = []

        # Creating Dataset

        rootname = "inputs/jazz_xlab/"
        filenames = os.listdir(rootname)
        dictChord, listChord = chordUtil.getDictChord(eval(alphabet))

        dataset = dataImport.ChordSeqDataset(
            filenames, rootname, alphabet, dictChord, sequence_lenght_in+sequence_lenght_out)

        # Create generators
        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': num_workers}

        if training:
            training_generator = data.DataLoader(dataset, **params)

        # TODO Put more optimisers
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate_discriminator)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=learning_rate_discriminator)
        else:
            raise ValueError("This optimizer is unknown to me")

        # TODO Put more losses
        if lossFunction == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        elif lossFunction == "MSE":
            criterion = nn.MSELoss()
        else:
            raise ValueError("This loss function is unknown to me")

        if training:
            print("Start training")

        for epoch in range(epochs):
            # Training
            if training:
                self.train(mode=True)
                for local_batch, local_labels in training_generator:
                    output, loss, disOnGenWinOne = self.oneBatchTrainOnGen(
                        local_batch, local_labels, optimizer, criterion, device, sequence_lenght_in, sequence_lenght_out, sampling, genNet)
                    total_loss += loss
                    disOnGenWin += disOnGenWinOne
                    totalSize += len(local_batch)

                if epoch % print_every == 0 and epoch != 0:
                    disOnGenWin = disOnGenWin/totalSize
                    total_loss = total_loss/totalSize
                    disOnGenWinTest.append(disOnGenWin)
                    all_losses.append(total_loss)
                    print('disOnGen : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (plotAndTimeUtil.timeSince(
                        start), epoch, epoch / epochs * 100, total_loss*100, disOnGenWin*100))
                    disOnGenWin = 0
                    total_loss = 0
                    totalSize = 0

            # Testing
            # self.train(mode=False)

        return all_losses, disOnGenWinTest


# define Train on one batch function


    def oneBatchTrainOnGen(self, local_batch, local_labels, optimizer, criterion, device, sequence_lenght_in, sequence_lenght_out, sampling, genNet):

        optimizer.zero_grad()
        loss = 0
        correct_guess = 0
        softmax = nn.Softmax(dim=1)
        temperature = 2

        # if tensor of shape 1 in loss function (ex : CrossEntropy)
        #local_labels_argmax = torch.tensor([torch.argmax(local_label) for local_label in local_labels]).to(device)

        #local_labels = local_labels.to(device)
        working_batch = torch.zeros(
            [len(local_batch), sequence_lenght_in+sequence_lenght_out, len(local_batch[0, 0])])
        working_batch[:, 0:sequence_lenght_in,
                      :] = local_batch[:, 0:sequence_lenght_in, :]
        local_batch = local_batch.to(device)

        for i in range(sequence_lenght_out):
            output = genNet.forward(local_batch)
            #local_batch[:,0:sequence_lenght_in-1,:] = local_batch[:,1:sequence_lenght_in,:]

            if sampling:
                #choice = choices(range(len(listChord)),softmax(output[0]))[0]
                # print(output.size())
                output = softmax(output)
                _output = output.cpu().div(temperature).exp().data

                # print(output.size())
                topi = torch.multinomial(_output, 1)
                # print(topi.size())
                # print(working_batch.size())
                for k in range(len(local_batch)):
                    working_batch[k, sequence_lenght_in +
                                  i, topi[k, 0].item()] = 1
                # print(working_batch[0])
            else:
                # TODO
                generated_sequence[sequence_lenght_in] = listChord[torch.argmax(
                    output_probability).item()]
                local_batch[:, sequence_lenght_in,
                            torch.argmax(output[:, :]).item()] = 1

        # print(working_batch.size())
        # local_batch has been transformed in output
        self.to(device)
        disDecision = self(working_batch[:, sequence_lenght_in:, :].to(device))

        # TODO : change that 0 in something more relevant
        fooling = 0
        for i in range(len(local_batch)):
            if disDecision[i, 0].item() < 0.5:
                fooling += 1
        # print(disDecision.size())
        loss = criterion(disDecision, torch.zeros(
            [len(local_batch), 1], dtype=torch.float).to(device))
        loss.backward()
        optimizer.step()

        return output, loss.item(), fooling

    def trainOnData(self, model_type, print_every, optimizer, lossFunction, alphabet='a0', sequence_lenght_in=16, sequence_lenght_out=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size_discriminator=128, num_layers_discriminator=2, dropout_discriminator=0.1, learning_rate_discriminator=1e-4, epochs=10, sampling=True):
        self.train(mode=True)
        if using_cuda:
            print("Trying using Cuda ...")
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                print("OK")
            else:
                print("Woops, Cuda cannot be found :'( ")
            device = torch.device("cuda:0" if use_cuda else "cpu")
        else:
            device = torch.device("cpu")
            print("Using Cpu")
        self.to(device)

        start = time.time()

        all_losses, similarity = self.doEpochsOnData(model_type, epochs, print_every, optimizer, lossFunction, learning_rate_discriminator,
                                                     batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, sampling, training=True)

        print("Finished Training")
        return similarity

    def doEpochsOnData(self, model_type, epochs, print_every, optimizer, lossFunction, learning_rate_discriminator, batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, sampling, training=True):

        # Init Training results and monitoring data
        all_losses = []
        total_loss = 0  # Reset every plot_every iters
        disOnDataWin = 0
        totalSize = 0
        disOnDataWinTest = []

        # Creating Dataset

        rootname = "inputs/jazz_xlab/"
        filenames = os.listdir(rootname)
        dictChord, listChord = chordUtil.getDictChord(eval(alphabet))

        dataset = dataImport.ChordSeqDataset(
            filenames, rootname, alphabet, dictChord, sequence_lenght_out)

        # Create generators
        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': num_workers}

        if training:
            training_generator = data.DataLoader(dataset, **params)

        # TODO Put more optimisers
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate_discriminator)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=learning_rate_discriminator)
        else:
            raise ValueError("This optimizer is unknown to me")

        # TODO Put more losses
        if lossFunction == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        elif lossFunction == "MSE":
            criterion = nn.MSELoss()
        else:
            raise ValueError("This loss function is unknown to me")

        if training:
            print("Start training")

        for epoch in range(epochs):
            # Training
            if training:
                self.train(mode=True)
                for local_batch, local_labels in training_generator:
                    loss, disOnDataWinOne = self.oneBatchTrainOnData(
                        local_batch, local_labels, optimizer, criterion, device, sequence_lenght_in, sequence_lenght_out, sampling)
                    total_loss += loss
                    disOnDataWin += disOnDataWinOne
                    totalSize += len(local_batch)

                if epoch % print_every == 0 and epoch != 0:
                    disOnDataWin = disOnDataWin/totalSize
                    total_loss = total_loss/totalSize
                    disOnDataWinTest.append(disOnDataWin)
                    all_losses.append(total_loss)
                    print('disOnDataWin : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (
                        plotAndTimeUtil.timeSince(start), epoch, epoch / epochs * 100, total_loss*100, disOnDataWin*100))
                    disOnDataWin = 0
                    total_loss = 0
                    totalSize = 0

            # Testing
            # self.train(mode=False)

        return all_losses, disOnDataWinTest


# define Train on one batch function


    def oneBatchTrainOnData(self, local_batch, local_labels, optimizer, criterion, device, sequence_lenght_in, sequence_lenght_out, sampling):

        optimizer.zero_grad()
        loss = 0
        correct_guess = 0
        softmax = nn.Softmax(dim=1)
        temperature = 2

        self.to(device)
        disDecision = self(local_batch.to(device))

        # TODO : change that 0 in something more relevant
        fooling = 0
        for i in range(len(local_batch)):
            if disDecision[i, 0].item() > 0.5:
                fooling += 1
        # print(disDecision.size())
        loss = criterion(disDecision, torch.ones(
            [len(local_batch), 1], dtype=torch.float).to(device))
        loss.backward()
        optimizer.step()

        return loss.item(), fooling
