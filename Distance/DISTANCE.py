# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:51:06 2018

@author: Paul
"""


import itertools
import numpy as np

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Données et Fonctions de base

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


#           1     2     3     4  5     6     7
Cmaj =     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
Cmin =     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
Caug =     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
Cdim =     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
Csus4 =    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
Csus2 =    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
C7 =       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Cmaj7 =    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
Cmin7 =    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
Cminmaj7 = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
Cmaj6 =    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
Cmin6 =    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
Cdim7 =    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
ChdimZ =   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
Cmaj9 =    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
Cmin9 =    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
C9 =       [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Cb9 =      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Cd9 =      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Cmin11 =   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
C11 =      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Cd11 =     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Cmaj13 =   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]
Cmin13 =   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
C13 =      [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
Cb13 =     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
C1 =       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C5 =       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def chord_to_chordNb(chord):
    
    chordNb=[]
    
    for i in range (0,12):
        if chord[i]==1:
            chordNb.append(i)
            
    return chordNb

#chord_to_chordNb(Cmaj)=[0,4,7]
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Définitions de tous les accords utiles

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

CHORDS_TO_STUDY = [Cmaj, Cmin]    #paramètre à changer
  
C = CHORDS_TO_STUDY

C = [chord_to_chordNb(i) for i in C]        #Cercle chromatique

C_quintes = [sorted([(7*i)%12 for i in j]) for j in C]   #Cercle des quintes


Cd = [sorted([(1+i)%12 for i in j]) for j in C] 
D  = [sorted([(2+i)%12 for i in j]) for j in C]
Dd = [sorted([(3+i)%12 for i in j]) for j in C] 
E  = [sorted([(4+i)%12 for i in j]) for j in C] 
F  = [sorted([(5+i)%12 for i in j]) for j in C] 
Fd = [sorted([(6+i)%12 for i in j]) for j in C] 
G  = [sorted([(7+i)%12 for i in j]) for j in C] 
Gd = [sorted([(8+i)%12 for i in j]) for j in C] 
A  = [sorted([(9+i)%12 for i in j]) for j in C] 
Ad = [sorted([(10+i)%12 for i in j]) for j in C] 
B  = [sorted([(11+i)%12 for i in j]) for j in C]

Cd_quintes = [sorted([(7*i)%12 for i in j]) for j in Cd]
D_quintes  = [sorted([(7*i)%12 for i in j]) for j in D]
Dd_quintes = [sorted([(7*i)%12 for i in j]) for j in Dd]
E_quintes  = [sorted([(7*i)%12 for i in j]) for j in E]
F_quintes  = [sorted([(7*i)%12 for i in j]) for j in F]
Fd_quintes = [sorted([(7*i)%12 for i in j]) for j in Fd]
G_quintes  = [sorted([(7*i)%12 for i in j]) for j in G]
Gd_quintes = [sorted([(7*i)%12 for i in j]) for j in Gd]
A_quintes  = [sorted([(7*i)%12 for i in j]) for j in A]
Ad_quintes = [sorted([(7*i)%12 for i in j]) for j in Ad]
B_quintes  = [sorted([(7*i)%12 for i in j]) for j in B]

TOTAL = C + Cd + D + Dd + E + F + Fd + G + Gd + A + Ad + B

TOTAL_Q = C_quintes+Cd_quintes+D_quintes+Dd_quintes+E_quintes+F_quintes+Fd_quintes+G_quintes+Gd_quintes+A_quintes+Ad_quintes+B_quintes

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Fonctions Distances entre 2 Accords

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#list(itertools.permutations([1, 2, 3])) = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
        
        
def distance_trql(chordNb1,chordNb2,p):        #Pour deux accords avec le mêmes nombre de notes 
    
    permutation = list(itertools.permutations(chordNb2))
    dist = []
    
    for i in range (0,len(permutation)):
        Sum = 0
        
        for j in range(0,len(chordNb1)) :
            Sum += min(((chordNb1[j] - permutation[i][j])%12)**p,((permutation[i][j] - chordNb1[j])%12)**p)
            Sum = Sum**(1/p)
            
        dist.append(Sum)
        
    return min(dist)
        
#distance_trql([0,4,7],[0,3,7]) = 1 

def add_notes(chord1,chord2):     #utile si les deux accords n'ont pas le même nb de notes
    
    result = [chord1]
    
    for k in range (len(chord2)-len(chord1)):
        temp=[]
        
        for l in result :
            
            for note in chord1 :
                temp.append(l+[note])
                
        result = temp
        
    return result
            
#add_notes([0,4,7],[0,4,7,10]) = [[0, 4, 7, 0], [0, 4, 7, 4], [0, 4, 7, 7]]
    
    
def distance(chord1,chord2,p=1):
    
    if len(chord1) == len(chord2):
        
        return distance_trql(chord1,chord2,p)
    
    if len(chord1) > len(chord2):
        
        temp = chord1
        chord1 = chord2
        chord2 = temp
    
    if len(chord1) < len(chord2):
        
        dist_ajout_de_notes=[]
        new_chord1 = add_notes(chord1,chord2)     #on va regarder toutes les possibilités possible
        
        for chord in new_chord1:
            dist_ajout_de_notes.append(distance_trql(chord,chord2,p))

        return min(dist_ajout_de_notes)

#distance([0, 4, 7] , [0, 4, 7, 10]) = 2


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Matrice de distance 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    


def matrice_dist(list_of_chords,p=1):
    n = len(list_of_chords)
    M = np.ones((n,n))
    for i in range (0,n) :
        for j in range (0,n) :
            M[i,j] = round(distance(list_of_chords[i],list_of_chords[j],p),2)
    return M
            
M1 = matrice_dist(TOTAL_Q,1)
M2 = matrice_dist(TOTAL_Q,2)
m2 = matrice_dist(TOTAL_Q,0.5)
M10 = matrice_dist(TOTAL_Q,10)



def tonalité():  
    tona = []
    M = matrice_dist(TOTAL_Q)
    tona.append(M[0,0]) #C
    tona.append(M[0,5]) #d
    tona.append(M[0,9]) #e
    tona.append(M[0,10]) #F
    tona.append(M[0,14]) #G
    tona.append(M[0,19]) #a
    tona.append(M[0,23]) #b
    print(tona)
    


def test():

	import random
	from random import randint
	for i in range(10000):
		a = randint(3,4)
		b = randint(3,4)
		c = randint(3,4)
		l1 = []
		l2 = []
		l3 = []
		for j in range(a):
			l1.append(randint(0,11))
		for j in range(b):
			l2.append(randint(0,11))
		for j in range(c):
			l3.append(randint(0,11))
		if i%1000:
			print(str(i)+" itérations et tout va bien")
		if int(distance(l1,l2)) > int(distance(l1,l3) + distance(l2,l3)):
			print("triste")
			return l1, l2, l3


#temp = temp[:-1]     #enlève le dernier élément de la liste


