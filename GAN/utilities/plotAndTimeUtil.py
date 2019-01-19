import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import matplotlib.colors as pltc
from random import sample


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

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
	plt.legend(loc='upper left', frameon=False)
	plt.plot(accuracy_test, label="test")
	plt.legend(loc='upper left', frameon=False)
	plt.show()


def PlotAllResults(data):
    superEpochs = len(data[0])-1
    all_colors = [k for k,v in pltc.cnames.items()]
    colorsTrain = sample(all_colors, superEpochs)
    colorsTest = sample(all_colors, superEpochs)
    print(colorsTrain)
    plt.figure()
    plt.title("Loss")

    counter = 0
    for i in range(1, superEpochs+1):
        plt.plot(range(counter,counter+len(data[3][i])),data[3][i], colorsTrain[i-1], label="train"+str(i)+" "+data[1][i]+" "+data[2][i]+" "+data[7][i]+" lr = "+str(data[8][i])+" Paul distance = "+str(data[9][i]))
        plt.plot(range(counter,counter+len(data[3][i])),data[4][i], colorsTest[i-1] , label="test"+str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        counter+=len(data[3][i])-1
    plt.show()


    plt.figure()
    plt.title("Accuracy")
    counter = 0
    for i in range(1, superEpochs+1):
        plt.plot(range(counter, counter+len(data[3][i])),data[5][i], colorsTrain[i-1], label="train"+str(i)+" "+data[1][i]+" "+data[2][i]+" "+data[7][i]+" lr = "+str(data[8][i])+" Paul distance = "+str(data[9][i]))
        plt.plot(range(counter, counter+len(data[3][i])),data[6][i], colorsTest[i-1], label="test"+str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        counter+=len(data[3][i])-1
    plt.show()
