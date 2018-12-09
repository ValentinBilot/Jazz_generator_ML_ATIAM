import numpy as np
import chordsDistances
import chordUtil
from chordsDistances import getPaulMatrix
import os



# only works for 'a0' for thre moment
def remapPaulToTristan(PaulBaseMatrix, alphabet):
    PaulChords=["C:maj","C:min","C#:maj","C#:min",
    "D:maj","D:min","D#:maj","D#:min",
    "E:maj","E:min","F:maj","F:min",
    "F#:maj","F#:min","G:maj","G:min"
    ,"G#:maj","G#:min","A:maj","A:min"
    ,"A#:maj","A#:min","B:maj","B:min","N"]

    rootname = "../inputs/jazz_xlab/"
    filenames = os.listdir(rootname)
    dictChord, listChord = chordUtil.getDictChord(eval(alphabet))


    TransfertBaseMatrix = np.zeros(shape = (len(PaulBaseMatrix),len(PaulBaseMatrix)))

    
    for i in range(len(PaulBaseMatrix)):
        TransfertBaseMatrix[i, dictChord[PaulChords[i]]] = 1
        
    TristanBaseMatrix = (TransfertBaseMatrix.T).dot(PaulBaseMatrix.dot(TransfertBaseMatrix))
        
    return TristanBaseMatrix

    
    
if __name__ == '__main__':
    M = chordsDistances.getPaulMatrix()
    print(remapPaulToTristan(M,"a0"))