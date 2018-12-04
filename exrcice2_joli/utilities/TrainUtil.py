import torch
from random import randint
from utilities import chordUtil
from utilities import dataImport
from utilities.chordUtil import *
from utilities.dataImport import *
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



def TrainModel(model, print_every, plot_every, optimizer, lossFunction, model_type="lstm", alphabet='a0', sequence_lenght=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size=128, num_layers=2, dropout=0.1, learning_rate=1e-4, epochs=10):

	# CUDA for PyTorch
	print("cuda available:")
	use_cuda = torch.cuda.is_available()
	print(use_cuda)
	use_cuda = using_cuda
	print("using cuda :")
	print(use_cuda)
	device = torch.device("cuda:0" if use_cuda else "cpu")

	# Load Model
	#model = LoadModel(model_type ,alphabet, sequence_lenght, hidden_size, num_layers, dropout, device)
	model.to(device)


	all_losses, test_losses, accuracy_train, accuracy_test = doEpochs(model, model_type, epochs, print_every, plot_every, optimizer, lossFunction, learning_rate, batch_size, shuffle, num_workers, alphabet, sequence_lenght, device)

	#SaveModel(model, model_type ,alphabet, sequence_lenght, hidden_size, num_layers, dropout)

	PlotResults(all_losses, test_losses, accuracy_train, accuracy_test)

	print("Tadam !")
	return





	
def PlotResults(all_losses, test_losses, accuracy_train, accuracy_test):
	plt.figure()
	plt.title("Loss")
	plt.plot(all_losses, label="train")
	plt.legend(loc='upper right', frameon=False)
	plt.plot(test_losses, label="test")
	plt.legend(loc='upper right', frameon=False)
	plt.show()

	plt.figure()
	plt.title("Accuracy")
	plt.plot(accuracy_train, label="train")
	plt.legend(loc='upper right', frameon=False)
	plt.plot(accuracy_test, label="test")
	plt.legend(loc='upper right', frameon=False)
	plt.show()




def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



def doEpochs(model, model_type, epochs, print_every, plot_every, optimizer, lossFunction, learning_rate, batch_size, shuffle, num_workers, alphabet, sequence_lenght, device):

	# Init Training results and monitoring data
	all_losses = []
	test_losses = []
	total_loss = 0 # Reset every plot_every iters
	test_loss = 0
	correct_guess_train, wrong_guess_train, correct_guess_test, wrong_guess_test = 0, 0, 0, 0
	accuracy_test = []
	accuracy_train = []
	start = time.time()

	rootname = "inputs/jazz_xlab/"
	filenames = os.listdir(rootname)
	dictChord, listChord = chordUtil.getDictChord(eval(alphabet))

	# Create datasets
	files_train ,files_test = train_test_split(filenames,test_size=0.7)
	dataset_train = dataImport.ChordSeqDataset(files_train, rootname, alphabet, dictChord, sequence_lenght)
	dataset_test = dataImport.ChordSeqDataset(files_test, rootname, alphabet, dictChord, sequence_lenght)

	# Create generators
	params = {'batch_size': batch_size,
		      'shuffle': shuffle,
		      'num_workers': num_workers}
	training_generator = data.DataLoader(dataset_train, **params)
	testing_generator = data.DataLoader(dataset_test, **params)

	#TODO Put more optimisers
	if optimizer == "Adam":
		optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	else:
		raise ValueError("This optimizer is unknown to me")
		
	#TODO Put more losses
	if lossFunction == "CrossEntropy":
		criterion = nn.CrossEntropyLoss()
	else:
		raise ValueError("This loss function is unknown to me")


	print("Start training")
	for epoch in range(1, epochs):
		# Training
		model.train(mode=True)
		for local_batch, local_labels in training_generator:
		    output, loss, correct_guess, wrong_guess = train(model, local_batch, local_labels, optimizer, criterion, device)
		    total_loss += loss
		    correct_guess_train += correct_guess
		    wrong_guess_train += wrong_guess

		if epoch % print_every == 0:
		    accuracy = correct_guess_train/(correct_guess_train+wrong_guess_train)
		    print('Train : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (timeSince(start), epoch, epoch / epochs * 100, loss, accuracy*100))

		if epoch % plot_every == 0:
		    accuracy = correct_guess_train/(correct_guess_train+wrong_guess_train)
		    all_losses.append(total_loss / (plot_every ))
		    accuracy_train.append(accuracy*100)
		    total_loss = 0
		    correct_guess_train, wrong_guess_train = 0, 0

		    

		# Testing
		model.train(mode=False)
		for local_batch, local_labels in testing_generator:
		    output, loss, correct_guess, wrong_guess  = test(model, local_batch, local_labels, optimizer, criterion, device)
		    test_loss +=loss
		    
		    correct_guess_test += correct_guess
		    wrong_guess_test += wrong_guess
		    

		if epoch % print_every == 0:
		    accuracy = correct_guess_test/(correct_guess_test+wrong_guess_test)
		    print('%s (%d %d%%) test, loss : %.4f, accuracy : %.4f%%' % (timeSince(start), epoch, epoch / epochs * 100, loss, accuracy*100))

		if epoch % plot_every == 0:
		    accuracy = correct_guess_test/(correct_guess_test+wrong_guess_test)
		    test_losses.append(test_loss / (plot_every ))
		    accuracy_test.append(accuracy*100)
		    test_loss = 0
		    correct_guess_test, wrong_guess_test = 0, 0

	print("Finished Training")
	return all_losses, test_losses, accuracy_train, accuracy_test
		    


def train(model, local_batch, local_labels, optimizer, criterion, device):
    optimizer.zero_grad()
    loss = 0
    correct_guess, wrong_guess = 0, 0

    # if tensor of shape 1 in loss function (ex : CrossEntropy)
    local_labels_argmax = torch.tensor([torch.argmax(local_label) for local_label in local_labels]).to(device)
    
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)    
    output = model(local_batch)
    
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




# Defining test function

def test(model, local_batch, local_labels, optimizer, criterion, device):
    loss = 0
    correct_guess, wrong_guess = 0, 0

    # if tensor of shape 1 in loss function (ex : CrossEntropy)
    local_labels_argmax = torch.tensor([torch.argmax(local_label) for local_label in local_labels]).to(device)  
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    output = model(local_batch)
    
    for i in range(len(local_batch)):
        if torch.argmax(output[i]) == torch.argmax(local_labels[i]):
            correct_guess += 1
        else:
            wrong_guess += 1
    loss = criterion(output, local_labels_argmax)

    return output, loss.item() / len(local_batch), correct_guess, wrong_guess
    
