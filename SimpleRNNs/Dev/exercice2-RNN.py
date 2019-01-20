import time
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import math
'''
LSTM realbook by Keunwoo Choi. (Keras 1.0)
Details on 
- repo:  https://github.com/keunwoochoi/lstm_real_book
- paper: https://arxiv.org/abs/1604.05358#
'''
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
#from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

import torch
import torch.nn as nn


# Still some cleaning TODO ...


# Importing dataset (already dyone in main)

# all_chords = #list of chords to write
# n_chords = len(all_chords) + 1 # Plus EOS marker


# Creating the network
# using RNN for the moment

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def vectorize(sentences, maxlen, num_chars, char_indices):
    max_dataset_lenght = 10000
    print('Vectorization...')
    len_dataset = min(max_dataset_lenght, len(sentences))
    X = torch.zeros(len_dataset, maxlen, 1, num_chars)
    Y = torch.zeros((len_dataset, maxlen, 1), dtype=torch.long)
    for i, sentence in enumerate(sentences[0:len_dataset]):
        for t, char in enumerate(sentence[0:len_dataset]):
            X[i, t, 0, char_indices[char]] = 1
            Y[i, t, 0] = char_indices[char]
    return X, Y


def train(X_in, Y_out):
    # target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(len(X_in) - 1):
        output, hidden = rnn(X_in[i], hidden)
        l = criterion(output, Y_out[i+1])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / X.size(0)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


character_mode = False

path = 'chord_sentences.txt'  # the txt data source
text = open(path).read()
print('corpus length:', len(text))

if character_mode:
    chars = set(text)
else:
    chord_seq = text.split(' ')
    chars = set(chord_seq)
    text = chord_seq

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
num_chars = len(char_indices)
print('total chars:', num_chars)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
sentences = []
#next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    #next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

# text to vectors
X, Y = vectorize(sentences, maxlen, num_chars, char_indices)


# build the model: stacked LSTM
#model = get_model(maxlen, num_chars)

rnn = RNN(num_chars, 128, num_chars)

n_iters = 100
print_every = 5
plot_every = 5
all_losses = []
total_loss = 0  # Reset every plot_every iters

start = time.time()

criterion = nn.NLLLoss()

learning_rate = 0.0005

for iter in range(1, n_iters + 1):
    index_in_data = iter % len(X)
    output, loss = train(X[index_in_data], Y[index_in_data])
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' %
              (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


######################################################################
# Plotting the Losses
# -------------------
#
# Plotting the historical loss from all\_losses shows the network
# learning:
#


plt.figure()
plt.plot(all_losses)
plt.show()

# Last Step is to create chords sequences from one or multiple starting chords.
