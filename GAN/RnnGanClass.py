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


class MYRNN(nn.Module):

    # Defining Pytorch things to make the Network

    def __init__(self, input_size, hidden_size_generator, num_layers_generator, dropout_generator):
        super(MYRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size_generator, num_layers = num_layers_generator, batch_first = True, dropout = dropout_generator)
        self.last_fully_connected = nn.Linear(hidden_size_generator, input_size)
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
        self.trainingData = [[0],["None"],["None"],[[0]],[[0]],[[0]],[[0]],[0],[0],["No"]]


    def forward(self, input_batch):
        output, hidden = self.rnn(input_batch)
        output = output[:,-1,:]
        print("outputsize : ")
        print(output.size())
        output = self.last_fully_connected(output)
        print("outputsize : ")
        print(output.size())
        # not sure about the dim = 1 in softmax
        output = self.softmax(output)
        print("outputsize : ")
        print(output.size())
        return output

    def trainOnDis(self, model_type, print_every, optimizer, lossFunction, data_set, disNet, alphabet='a0', sequence_lenght_in=16, sequence_lenght_out=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size_generator=128, num_layers_generator=2, dropout_generator=0.1, learning_rate_generator=1e-4, epochs=10, sampling=True):
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

        all_losses, similarity = self.doEpochs(model_type, epochs, print_every, optimizer, lossFunction, learning_rate_generator, batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, data_set, disNet, sampling, training=True)

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

    def doEpochs(self, model_type, epochs, print_every, optimizer, lossFunction, learning_rate_generator, batch_size, shuffle, num_workers, alphabet, sequence_lenght_in, sequence_lenght_out, device, start, data_set, disNet, sampling, training=True):

        # Init Training results and monitoring data
        all_losses = []
        total_loss = 0 # Reset every plot_every iters
        genWin = 0
        totalSize = 0
        genWinTest = []

        # Creating Dataset
            # dataset has been passed in the parameters
        
        # Create generators
        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': num_workers}
            
        if training:
            training_generator = data.DataLoader(dataset, **params)

        #TODO Put more optimisers
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        else:
            raise ValueError("This optimizer is unknown to me")

        #TODO Put more losses
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
                    output, loss, genWinOne = self.oneBatchTrain(local_batch, local_labels, optimizer, criterion, device, sampling)
                    total_loss += loss
                    genWin += genWinOne
                    totalSize += len(local_batch)

                if epoch % print_every == 0 and epoch != 0:
                    genWin = genWin/totalSize
                    total_loss = total_loss/totalSize
                    genWinTest.append(genWin)
                    all_losses.append(total_loss)
                    print('genWin : %s (%d %d%%) loss : %.4f, accuracy : %.4f%%' % (plotAndTimeUtil.timeSince(start), epoch, epoch / epochs * 100, total_loss, genWin*100))
                    genWin = 0
                    total_loss = 0
                    
            # Testing
            #self.train(mode=False)

        return all_losses, genWinTest



# define Train on one batch function

    def oneBatchTrain(self, local_batch, local_labels, optimizer, criterion, device, sequence_lenght_in sequence_lenght_out, sampling):
       
        optimizer.zero_grad()
        loss = 0
        correct_guess = 0
        softmax = nn.Softmax(dim=1)


        # if tensor of shape 1 in loss function (ex : CrossEntropy)
        #local_labels_argmax = torch.tensor([torch.argmax(local_label) for local_label in local_labels]).to(device)
        
# THIS PART IS NOT FINISHED AT ALLLLLL

        local_batch = local_batch.to(device)
        #local_labels = local_labels.to(device)
        for i in range(sequence_lenght_out):
            output = self.forward(local_batch)
            local_batch[:,0:sequence_lenght_in-1,:] = local_batch[:,1:sequence_lenght_in,:]
            
            if sampling:
                choice = choices(range(len(listChord)),softmax(output[0]))[0]
                local_batch[:,sequence_lenght_in, choice] = 1
            else:
                generated_sequence[sequence_lenght_in] = listChord[torch.argmax(output_probability).item()]
                local_batch[:, sequence_lenght_in, torch.argmax(output[:,:]).item()] = 1                
                
        
        for i in range(len(local_batch)):
            if torch.argmax(output[i]) == torch.argmax(local_labels[i]):
                correct_guess += 1
            else:
                wrong_guess += 1
                

        loss = loss_mult_coeff * criterion(output, local_labels_argmax)
        loss.backward()
        optimizer.step()

        return output, loss.item(), correct_guess



    def plotLastTraining(self):
        plotAndTimeUtil.PlotResults(self.trainingData[3][-1], self.trainingData[4][-1], self.trainingData[5][-1], self.trainingData[6][-1])

    def plotAllTraining(self):
        plotAndTimeUtil.PlotAllResults(self.trainingData)

    def toString(self, model_type, print_every, plot_every, optimizer, lossFunction, alphabet='a0', sequence_lenght=16, using_cuda=True, batch_size=128, shuffle=True, num_workers=6, hidden_size=128, num_layers=2, dropout=0.1, learning_rate=1e-4, epochs=10, use_Paul_distance=False):
        #dropoutstr = str(dropout).replace('.',',')
        model_string = "models/"+model_type+str(num_layers)+"layers"+str(hidden_size)+"blocks"+alphabet+"alphabet"+str(sequence_lenght)+"lenSeq.pt"
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
        test_sequence_tensor = torch.zeros(1, len(test_sequence), len(dictChord)).to(device)
        last_chords_output = torch.zeros(1, lenSeq, len(dictChord)).to(device)
        test_sequence_tensor.requires_grad = False
        last_chords_output.requires_grad = False
        for t in range(len(test_sequence)):
            test_sequence_tensor[0, t, dictChord[test_sequence[t]]] = 1
            if t != len(test_sequence)-1 :
                last_chords_output[0, t-1, dictChord[test_sequence[t]]] = 1

                
        generated_sequence = [0 for i in range(generation_lenght)]
        generated_sequence[0:lenSeq] = test_sequence

        self.train(mode=False)
        softmax = nn.Softmax(dim=0)

        for t in range(generation_lenght-lenSeq):
            if t == 0:
                output_probability = self(test_sequence_tensor)

                if sampling:
                    choice = choices(range(len(listChord)),softmax(output_probability[0]))[0]
                    generated_sequence[t+lenSeq] = listChord[choice]
                    last_chords_output[0, lenSeq-1, choice] = 1

                else: 
                    generated_sequence[t+lenSeq] = listChord[torch.argmax(output_probability).item()]
                    last_chords_output[0, lenSeq-1, torch.argmax(output_probability).item()] = 1


            else:

                last_chords_output.to(device)        
                output_probability = self(last_chords_output)
                last_chords_output[0,0:lenSeq-1] = last_chords_output[0,1:lenSeq]

                if sampling:
                    choice = choices(range(len(listChord)),softmax(output_probability[0]))[0]
                    generated_sequence[t+lenSeq] = listChord[choice]
                    last_chords_output[0, lenSeq-1, choice] = 1

                else:
                    last_chords_output[0, lenSeq-1, torch.argmax(output_probability).item()] = 1
                    generated_sequence[t+lenSeq] = listChord[torch.argmax(output).item()]


        for i in range(generation_lenght):
            if i%4 == 0:
                print(generated_sequence[i:i+4])
            if i == lenSeq-1 :
                print("generated :")
                
        if silent:
            return
        else:
            return generated_sequence
        

        