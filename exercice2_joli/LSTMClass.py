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

class MYLSTM(nn.Module):

    # Defining Pytorch things to make the Network

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(MYLSTM, self).__init__()
        self.gru = nn.LSTM(input_size, hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.last_fully_connected = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)

        # Usefull for monitoring
        self.trainingData = [[0],["None"],["None"],[[0]],[[0]],[[0]],[[0]],[0]]


    def forward(self, input_batch):
        output, (hidden, cell_state) = self.gru(input_batch)
        output = output[:,-1,:]
        output = self.last_fully_connected(output)
        output = self.softmax(output)
        return output

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

    def trainAndTest(self, model_type, print_every, plot_every, optimizer, lossFunction, alphabet='a0', sequence_lenght=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size=128, num_layers=2, dropout=0.1, learning_rate=1e-4, epochs=10):
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

        all_losses, test_losses, accuracy_train, accuracy_test = self.doEpochs(model_type, epochs, print_every, plot_every, optimizer, lossFunction, learning_rate, batch_size, shuffle, num_workers, alphabet, sequence_lenght, device, start, training=True)

        self.trainingData[0].append(self.trainingData[0][-1]+1)
        self.trainingData[1].append(optimizer)
        self.trainingData[2].append(lossFunction)
        self.trainingData[3].append(all_losses)
        self.trainingData[4].append(test_losses)
        self.trainingData[5].append(accuracy_train)
        self.trainingData[6].append(accuracy_test)
        self.trainingData[7].append(plotAndTimeUtil.timeSince(start))
        print("Finished Training")


        return

    def doEpochs(self, model_type, epochs, print_every, plot_every, optimizer, lossFunction, learning_rate, batch_size, shuffle, num_workers, alphabet, sequence_lenght, device, start, training=True):

    	# Init Training results and monitoring data
    	all_losses = []
    	test_losses = []
    	total_loss = 0 # Reset every plot_every iters
    	test_loss = 0
    	correct_guess_train, wrong_guess_train, correct_guess_test, wrong_guess_test = 0, 0, 0, 0
    	accuracy_test = []
    	accuracy_train = []

    	rootname = "inputs/jazz_xlab/"
    	filenames = os.listdir(rootname)
    	dictChord, listChord = chordUtil.getDictChord(eval(alphabet))

    	# Create datasets
    	files_train ,files_test = train_test_split(filenames,test_size=0.7)
    	if training:
    		dataset_train = dataImport.ChordSeqDataset(files_train, rootname, alphabet, dictChord, sequence_lenght)
    	dataset_test = dataImport.ChordSeqDataset(files_test, rootname, alphabet, dictChord, sequence_lenght)

    	# Create generators
    	params = {'batch_size': batch_size,
    		      'shuffle': shuffle,
    		      'num_workers': num_workers}

    	if training:
    		training_generator = data.DataLoader(dataset_train, **params)
    	testing_generator = data.DataLoader(dataset_test, **params)

    	#TODO Put more optimisers
    	if optimizer == "Adam":
    		optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
    	else:
    		raise ValueError("This optimizer is unknown to me")

    	#TODO Put more losses
    	if lossFunction == "CrossEntropy":
    		criterion = nn.CrossEntropyLoss()
    	else:
    		raise ValueError("This loss function is unknown to me")

    	if training:
    		print("Start training")
    	for epoch in range(1, epochs):
    		# Training
    		if training:
    			self.train(mode=True)
    			for local_batch, local_labels in training_generator:
    				output, loss, correct_guess, wrong_guess = self.oneBatchTrain(local_batch, local_labels, optimizer, criterion, device)
    				total_loss += loss
    				correct_guess_train += correct_guess
    				wrong_guess_train += wrong_guess

    			if epoch % print_every == 0:
    				accuracy = correct_guess_train/(correct_guess_train+wrong_guess_train)
    				print('Train : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (plotAndTimeUtil.timeSince(start), epoch, epoch / epochs * 100, loss, accuracy*100))

    			if epoch % plot_every == 0:
    				accuracy = correct_guess_train/(correct_guess_train+wrong_guess_train)
    				all_losses.append(total_loss / (plot_every ))
    				accuracy_train.append(accuracy*100)
    				total_loss = 0
    				correct_guess_train, wrong_guess_train = 0, 0



    		# Testing
    		self.train(mode=False)
    		for local_batch, local_labels in testing_generator:
    		    output, loss, correct_guess, wrong_guess  = self.oneBatchTest(local_batch, local_labels, optimizer, criterion, device)
    		    test_loss +=loss

    		    correct_guess_test += correct_guess
    		    wrong_guess_test += wrong_guess


    		if epoch % print_every == 0:
    			accuracy = correct_guess_test/(correct_guess_test+wrong_guess_test)
    			if training:
    				print('Test : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (plotAndTimeUtil.timeSince(start), epoch, epoch / epochs * 100, loss, accuracy*100))

    		if epoch % plot_every == 0:
    		    accuracy = correct_guess_test/(correct_guess_test+wrong_guess_test)
    		    test_losses.append(test_loss / (plot_every ))
    		    accuracy_test.append(accuracy*100)
    		    test_loss = 0
    		    correct_guess_test, wrong_guess_test = 0, 0

    	if training:
    		return all_losses, test_losses, accuracy_train, accuracy_test
    	else:
    		return test_losses, accuracy_test



# define Train on one batch function

    def oneBatchTrain(self, local_batch, local_labels, optimizer, criterion, device):
        optimizer.zero_grad()
        loss = 0
        correct_guess, wrong_guess = 0, 0

        # if tensor of shape 1 in loss function (ex : CrossEntropy)
        local_labels_argmax = torch.tensor([torch.argmax(local_label) for local_label in local_labels]).to(device)

        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        output = self.forward(local_batch)

        for i in range(len(local_batch)):
            #print(output[i].size(), local_labels[i].size())
            if torch.argmax(output[i]) == torch.argmax(local_labels[i]):
                correct_guess += 1
            else:
                wrong_guess += 1

        loss = criterion(output, local_labels_argmax)
        loss.backward()
        optimizer.step()

        return output, loss.item() / len(local_batch), correct_guess, wrong_guess


# Defining test on one batch function

    def oneBatchTest(self, local_batch, local_labels, optimizer, criterion, device):
        loss = 0
        correct_guess, wrong_guess = 0, 0

        # if tensor of shape 1 in loss function (ex : CrossEntropy)
        local_labels_argmax = torch.tensor([torch.argmax(local_label) for local_label in local_labels]).to(device)
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        output = self.forward(local_batch)

        for i in range(len(local_batch)):
            if torch.argmax(output[i]) == torch.argmax(local_labels[i]):
                correct_guess += 1
            else:
                wrong_guess += 1
        loss = criterion(output, local_labels_argmax)

        return output, loss.item() / len(local_batch), correct_guess, wrong_guess



    def plotLastTraining(self):
        plotAndTimeUtil.PlotResults(self.trainingData[3][-1], self.trainingData[4][-1], self.trainingData[5][-1], self.trainingData[6][-1])

    def plotAllTraining(self):
        plotAndTimeUtil.PlotAllResults(self.trainingData)

    def toString(self, model_type, print_every, plot_every, optimizer, lossFunction, alphabet='a0', sequence_lenght=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size=128, num_layers=2, dropout=0.1, learning_rate=1e-4, epochs=10):
        dropoutstr = str(dropout).replace('.',',')
        model_string = "models/"+model_type+str(num_layers)+"layers"+str(hidden_size)+"blocks"+alphabet+"alphabet"+str(sequence_lenght)+"lenSeq"+dropoutstr+"dropout.pt"
        return model_string