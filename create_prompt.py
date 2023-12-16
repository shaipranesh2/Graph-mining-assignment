import pandas as pd
import numpy as np
import sys

doc_content_list = []
# f = open('text_gcn/data/mr/text_train.txt', 'rb')
f = open('text_gcn/data/mr/text_train.txt', 'rb')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))
f.close()

getWgt = lambda x: (1 if x<100 else 0, doc_content_list[x])
getWgt2 = lambda x: ('positive' if x<100 else 'negative', doc_content_list[x])

def get0shot():
    dtpt = getWgt(np.random.randint(0,180))
    str = f'Movie review: {dtpt[1]}\nTask: Classify this review into one of two categories positive or negative. Respond with 0 for negative and 1 for positive. Do not ask more questions.'
    return (dtpt[0], str)

def get1shot(rands = None, shots:int = 3):
    if rands is None:
        rands = np.random.randint(0,180,shots)
        while 1:
            temp = np.random.randint(0,180)
            if temp not in rands:
                rands = np.append(rands, temp)
                break
    else:
        shots = len(rands)-1
    dtpts = [getWgt2(rand) for rand in rands]
    genMovieStr = lambda x: f'Movie review {x+1}: {dtpts[x][1]}\n'
    genStructStr = lambda x: f'Review {x+1} is classified as {dtpts[x][0]}'
    str = ''.join([genMovieStr(i) for i in range(shots)]) + ", ".join([genStructStr(i) for i in range(shots)]) + f'\nTask: Classify the following review review into one of two categories positive or negative. Respond with 0 for negative and 1 for positive. Do not ask more questions.\nMovie review: {dtpts[shots][1]}'
    return (dtpts[shots][0], str)

def getneigh(neighs:int = 3, rand:int = None, rands = None):
    if rands is None:
        if rand is None:
            rand = np.random.randint(0,2)
        if rand == 0:
            rands = np.random.randint(100,180,neighs)
        else:
            rands = np.random.randint(0,100,neighs)
    dtpts = [getWgt(rand) for rand in rands]
    genMovieStr = lambda x: f'Movie review {x+1}: {dtpts[x][1]}\n'
    str = ''.join([genMovieStr(i) for i in range(neighs)]) + f'Task: Please summarize the reviews above with a short paragraph, find some common points which can reflect the category of this reviews, given that they all belong to the same class. Do not ask more questions.'
    return (rand, str)

def getneigh2(neighs_sum:str, rand:int):
    dtpt = getWgt(np.random.randint(0,100) if rand == 1 else np.random.randint(100,180))
    str = f'Movie review: {dtpt[1]}\nNeighbour summary: {neighs_sum}\nTask: Classify this review into one of two categories positive or negative. Respond with 0 for negative and 1 for positive. Do not ask more questions.'
    return (dtpt[0], str)