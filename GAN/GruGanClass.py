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


class MYGRU(nn.Module):

    # Defining Pytorch things to make the Network

    def __init__(self, input_size, hidden_size_generator, num_layers_generator, dropout_generator):
        super(MYGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size_generator,
                          num_layers=num_layers_generator, batch_first=True, dropout=dropout_generator)
        self.last_fully_connected = nn.Linear(
            hidden_size_generator, input_size)
        # not sure about the dim = 1 in softmax
        self.softmax = nn.LogSoftmax(dim=1)

    # Defining some cool functions to make work easier and trop stylé

    # About the trainingData :
    # 0 is training number
    # 1 is optimizer used
    # 2 is loss used
    # 3 is loss values on train set
    # 4 is loss values on test set
    # 5 is accuracy on train set
    # 6 is accuracy on test set
    # 7 is time
    # 8 is learning rate
    # 9 is use_Paul_distance

        # Usefull for monitoring
        #TODO reimplement the monitoring part adapted to the GAN case
        self.trainingData = [[0], ["None"], ["None"], [
            [0]], [[0]], [[0]], [[0]], [0], [0], ["No"]]

    def forward(self, input_batch):
        output, hidden = self.rnn(input_batch)
        output = output[:, -1, :]
        output = self.last_fully_connected(output)
        # not sure about the dim = 1 in softmax
        output = self.softmax(output)
        return output

    def trainAndTest(self, model_type, print_every, plot_every, optimizer, lossFunction, alphabet='a0', sequence_lenght=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size=128, num_layers=2, dropout=0.1, learning_rate=1e-4, epochs=10, use_Paul_distance=False):
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

        all_losses, test_losses, accuracy_train, accuracy_test = self.doEpochs(
            model_type, epochs, print_every, plot_every, optimizer, lossFunction, learning_rate, batch_size, shuffle, num_workers, alphabet, sequence_lenght, device, start, use_Paul_distance, training=True)

        self.trainingData[0].append(self.trainingData[0][-1]+1)
        self.trainingData[1].append(optimizer)
        self.trainingData[2].append(lossFunction)
        self.trainingData[3].append(all_losses)
        self.trainingData[4].append(test_losses)
        self.trainingData[5].append(accuracy_train)
        self.trainingData[6].append(accuracy_test)
        self.trainingData[7].append(plotAndTimeUtil.timeSince(start))
        self.trainingData[8].append(learning_rate)
        self.trainingData[9].append(use_Paul_distance)

        print("Finished Training")

        return

    def doEpochs(self, model_type, epochs, print_every, plot_every, optimizer, lossFunction, learning_rate, batch_size, shuffle, num_workers, alphabet, sequence_lenght, device, start, use_Paul_distance, training=True):

        # Init Training results and monitoring data
        all_losses = []
        test_losses = []
        total_loss = 0  # Reset every plot_every iters
        test_loss = 0
        correct_guess_train, wrong_guess_train, correct_guess_test, wrong_guess_test = 0, 0, 0, 0
        accuracy_test = []
        accuracy_train = []

        rootname = "inputs/jazz_xlab/"
        filenames = os.listdir(rootname)
        dictChord, listChord = chordUtil.getDictChord(eval(alphabet))

        # Create datasets
        files_train, files_test = train_test_split(filenames, test_size=0.7)
        if training:
            dataset_train = dataImport.ChordSeqDataset(
                files_train, rootname, alphabet, dictChord, sequence_lenght)
        dataset_test = dataImport.ChordSeqDataset(
            files_test, rootname, alphabet, dictChord, sequence_lenght)

        # Create generators
        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': num_workers}

        # Get Paul distance matrix :
        if use_Paul_distance:
            M = chordsDistances.getPaulMatrix()
            M = remapChordsToBase.remapPaulToTristan(M, alphabet)
            distance_tensor = torch.tensor(M, dtype=torch.float).to(device)
        else:
            distance_tensor = 0

        if training:
            training_generator = data.DataLoader(dataset_train, **params)
        testing_generator = data.DataLoader(dataset_test, **params)

        # TODO Put more optimisers
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("This optimizer is unknown to me")

        # TODO Put more losses
        if lossFunction == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        if lossFunction == "MSE":
            criterion = nn.MSELoss()
        else:
            raise ValueError("This loss function is unknown to me")

            
        # Training
        if training:
            print("Start training")
        for epoch in range(1, epochs):
            if training:
                self.train(mode=True)
                for local_batch, local_labels in training_generator:
                    output, loss, correct_guess, wrong_guess = self.oneBatchTrain(
                        local_batch, local_labels, optimizer, criterion, device, use_Paul_distance, distance_tensor)
                    total_loss += loss
                    correct_guess_train += correct_guess
                    wrong_guess_train += wrong_guess

                if epoch % print_every == 0:
                    accuracy = correct_guess_train / \
                        (correct_guess_train+wrong_guess_train)
                    print('Train : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (
                        plotAndTimeUtil.timeSince(start), epoch, epoch / epochs * 100, loss, accuracy*100))

                if epoch % plot_every == 0:
                    accuracy = correct_guess_train / \
                        (correct_guess_train+wrong_guess_train)
                    all_losses.append(total_loss / (plot_every))
                    accuracy_train.append(accuracy*100)
                    total_loss = 0
                    correct_guess_train, wrong_guess_train = 0, 0

            # Testing
            self.train(mode=False)
            for local_batch, local_labels in testing_generator:
                output, loss, correct_guess, wrong_guess = self.oneBatchTest(
                    local_batch, local_labels, optimizer, criterion, device, use_Paul_distance, distance_tensor)
                test_loss += loss

                correct_guess_test += correct_guess
                wrong_guess_test += wrong_guess

            if epoch % print_every == 0:
                accuracy = correct_guess_test / \
                    (correct_guess_test+wrong_guess_test)
                if training:
                    print('Test : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (
                        plotAndTimeUtil.timeSince(start), epoch, epoch / epochs * 100, loss, accuracy*100))

            if epoch % plot_every == 0:
                accuracy = correct_guess_test / \
                    (correct_guess_test+wrong_guess_test)
                test_losses.append(test_loss / (plot_every))
                accuracy_test.append(accuracy*100)
                test_loss = 0
                correct_guess_test, wrong_guess_test = 0, 0

        if training:
            return all_losses, test_losses, accuracy_train, accuracy_test
        else:
            return test_losses, accuracy_test


# define Train on one batch function

    def oneBatchTrain(self, local_batch, local_labels, optimizer, criterion, device, use_Paul_distance, distance_tensor):
        optimizer.zero_grad()
        loss = 0
        correct_guess, wrong_guess = 0, 0

        # if tensor of shape 1 in loss function (ex : CrossEntropy) use the line bellow instead
        # local_labels_argmax = torch.tensor(
        #    [torch.argmax(local_label) for local_label in local_labels]).to(device)

        local_batch, local_labels = local_batch.to(
            device), local_labels.to(device)
        output = self.forward(local_batch)

        for i in range(len(local_batch)):
            if torch.argmax(output[i]) == torch.argmax(local_labels[i]):
                correct_guess += 1
            else:
                wrong_guess += 1

        # using Paul Distance
        if use_Paul_distance:
            loss_mult_coeff = 0
            for i in range(len(local_batch)):
                loss_mult_coeff += torch.matmul(distance_tensor, local_labels[i])[
                    torch.argmax(output[i])]

            loss_mult_coeff = loss_mult_coeff/len(local_batch)
        else:
            loss_mult_coeff = 1

        loss = loss_mult_coeff * criterion(output, local_labels)
        loss.backward()
        optimizer.step()

        return output, loss.item() / len(local_batch), correct_guess, wrong_guess


# Defining test on one batch function

    def oneBatchTest(self, local_batch, local_labels, optimizer, criterion, device, use_Paul_distance, distance_tensor):
        loss = 0
        correct_guess, wrong_guess = 0, 0

        # if tensor of shape 1 in loss function (ex : CrossEntropy) use the line bellow instead
        # local_labels_argmax = torch.tensor(
        #    [torch.argmax(local_label) for local_label in local_labels]).to(device)
        local_batch, local_labels = local_batch.to(
            device), local_labels.to(device)
        output = self.forward(local_batch)

        for i in range(len(local_batch)):
            if torch.argmax(output[i]) == torch.argmax(local_labels[i]):
                correct_guess += 1
            else:
                wrong_guess += 1

        # using Paul Distance
        if use_Paul_distance:
            loss_mult_coeff = 0
            for i in range(len(local_batch)):
                loss_mult_coeff += torch.matmul(distance_tensor, local_labels[i])[
                    torch.argmax(output[i])]

            loss_mult_coeff = loss_mult_coeff/len(local_batch)
        else:
            loss_mult_coeff = 1

        loss = criterion(output, local_labels)

        return output, loss.item() / len(local_batch), correct_guess, wrong_guess

    def trainOnDis(self, model_type, print_every, optimizer, lossFunction, disNet, alphabet='a0', sequence_lenght_in=16, sequence_lenght_out=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size_generator=128, num_layers_generator=2, dropout_generator=0.1, learning_rate_generator=1e-4, epochs=10, sampling=True):
        
        #TODO: understand why disNet.train(mode=False) makes the traniing fail
        # disNet.train(mode=False)
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

        all_losses, similarity = self.doEpochsOnDis(model_type, epochs, print_every, optimizer, lossFunction, learning_rate_generator,
                                                    batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, disNet, sampling, training=True)

        print("Finished Training")
        return similarity

    def doEpochsOnDis(self, model_type, epochs, print_every, optimizer, lossFunction, learning_rate_generator, batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, disNet, sampling, training=True):

        # Init Training results and monitoring data
        all_losses = []
        total_loss = 0  # Reset every plot_every iters
        genWin = 0
        totalSize = 0
        genWinTest = []

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
                self.parameters(), lr=learning_rate_generator)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=learning_rate_generator)
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
                    output, loss, genWinOne = self.oneBatchTrainOnDis(
                        local_batch, local_labels, optimizer, criterion, device, sequence_lenght_in, sequence_lenght_out, sampling, disNet)
                    total_loss += loss
                    genWin += genWinOne
                    totalSize += len(local_batch)

                if epoch % print_every == 0 and epoch != 0:
                    genWin = genWin/totalSize
                    total_loss = total_loss/totalSize
                    genWinTest.append(genWin)
                    all_losses.append(total_loss)
                    print('genWin : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (
                        plotAndTimeUtil.timeSince(start), epoch, epoch / epochs * 100, 100*total_loss, genWin*100))
                    genWin = 0
                    total_loss = 0
                    totalSize = 0

            # Testing
            # TODO implement testing
            # self.train(mode=False)

        return all_losses, genWinTest


# define Train on one batch function

    def oneBatchTrainOnDis(self, local_batch, local_labels, optimizer, criterion, device, sequence_lenght_in, sequence_lenght_out, sampling, disNet):

        optimizer.zero_grad()
        loss = 0
        correct_guess = 0
        softmax = nn.Softmax(dim=1)
        
        #TODO: put temperature in the parameters
        # temperature is for the sampling part
        # high temperature will increase likelyhood of unlikely events and reduce likelyhood of very probable events
        temperature = 2

        # if tensor of shape 1 in loss function (ex : CrossEntropy)
        #local_labels_argmax = torch.tensor([torch.argmax(local_label) for local_label in local_labels]).to(device)

        # initialasing the working_batch that will be used as a buffer for the generative part
        working_batch = torch.zeros(
            [len(local_batch), sequence_lenght_in+sequence_lenght_out, len(local_batch[0, 0])])
        working_batch[:, 0:sequence_lenght_in,
                      :] = local_batch[:, 0:sequence_lenght_in, :]
        local_batch = local_batch.to(device)

        for i in range(sequence_lenght_out):
            output = self.forward(local_batch)


            if sampling:
                #choice = choices(range(len(listChord)),softmax(output[0]))[0]
                output = softmax(output)
                _output = output.cpu().div(temperature).exp().data

                # doing the sampling
                topi = torch.multinomial(_output, 1)
                for k in range(len(local_batch)):
                    working_batch[k, sequence_lenght_in +
                                  i, topi[k, 0].item()] = 1
            else:
                # TODO
                generated_sequence[sequence_lenght_in] = listChord[torch.argmax(
                    output_probability).item()]
                local_batch[:, sequence_lenght_in,
                            torch.argmax(output[:, :]).item()] = 1

        
        disNet.to(device)
        # Getting discriminator output to compute the loss
        disDecision = disNet(
            working_batch[:, sequence_lenght_in:, :].to(device))

        # computing fooling (a.k.a accuracy)
        fooling = 0
        for i in range(len(local_batch)):
            if disDecision[i, 0].item() > 0.5:
                fooling += 1
                
        # Loss computing and backpropagation
        loss = criterion(disDecision, torch.ones(
            [len(local_batch), 1], dtype=torch.float).to(device))
        loss.backward()
        optimizer.step()

        return output, loss.item(), fooling

    def plotLastTraining(self):
        plotAndTimeUtil.PlotResults(
            self.trainingData[3][-1], self.trainingData[4][-1], self.trainingData[5][-1], self.trainingData[6][-1])

    def plotAllTraining(self):
        plotAndTimeUtil.PlotAllResults(self.trainingData)

    def toString(self, model_type, print_every, plot_every, optimizer, lossFunction, alphabet='a0', sequence_lenght=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size=128, num_layers=2, dropout=0.1, learning_rate=1e-4, epochs=10, use_Paul_distance=False):
        #dropoutstr = str(dropout).replace('.',',')
        model_string = "models/"+model_type+str(num_layers)+"layers"+str(
            hidden_size)+"blocks"+alphabet+"alphabet"+str(sequence_lenght)+"lenSeq.pt"
        return model_string

    def generateFromSequence(self, test_sequence, generation_lenght, alphabet, sampling=False, using_cuda=True, silent=True):
        lenSeq = len(test_sequence)

        # Cuda blabla
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

        # Getting chords dictionary
        rootname = "inputs/jazz_xlab/"
        filenames = os.listdir(rootname)
        dictChord, listChord = chordUtil.getDictChord(eval(alphabet))

        # Initialising objects
        test_sequence_tensor = torch.zeros(
            1, len(test_sequence), len(dictChord)).to(device)
        last_chords_output = torch.zeros(1, lenSeq, len(dictChord)).to(device)
        test_sequence_tensor.requires_grad = False
        last_chords_output.requires_grad = False
        for t in range(len(test_sequence)):
            test_sequence_tensor[0, t, dictChord[test_sequence[t]]] = 1
            if t != len(test_sequence)-1:
                last_chords_output[0, t-1, dictChord[test_sequence[t]]] = 1

        generated_sequence = [0 for i in range(generation_lenght)]
        generated_sequence[0:lenSeq] = test_sequence

        self.train(mode=False)
        softmax = nn.Softmax(dim=0)

        for t in range(generation_lenght-lenSeq):
            if t == 0:
                output_probability = self(test_sequence_tensor)

                if sampling:
                    choice = choices(range(len(listChord)),
                                     softmax(output_probability[0]))[0]
                    generated_sequence[t+lenSeq] = listChord[choice]
                    last_chords_output[0, lenSeq-1, choice] = 1

                else:
                    generated_sequence[t +
                                       lenSeq] = listChord[torch.argmax(output_probability).item()]
                    last_chords_output[0, lenSeq-1,
                                       torch.argmax(output_probability).item()] = 1

            else:

                last_chords_output.to(device)
                output_probability = self(last_chords_output)
                last_chords_output[0, 0:lenSeq -
                                   1] = last_chords_output[0, 1:lenSeq]

                if sampling:
                    choice = choices(range(len(listChord)),
                                     softmax(output_probability[0]))[0]
                    generated_sequence[t+lenSeq] = listChord[choice]
                    last_chords_output[0, lenSeq-1, choice] = 1

                else:
                    last_chords_output[0, lenSeq-1,
                                       torch.argmax(output_probability).item()] = 1
                    generated_sequence[t +
                                       lenSeq] = listChord[torch.argmax(output).item()]

        for i in range(generation_lenght):
            if i % 4 == 0:
                print(generated_sequence[i:i+4])
            if i == lenSeq-1:
                print("generated :")

        if silent:
            return
        else:
            return generated_sequence
