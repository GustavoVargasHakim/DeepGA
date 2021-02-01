# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:11:23 2020

@author: user
"""

import random
import math
from copy import deepcopy
from WangEncoding import *

def crossover(x, y):
    x = deepcopy(x)
    y = deepcopy(y)
    
    '''First parent'''
    x_nblock = x.n_block
    x_nfull = x.n_full
    xblocks = x.first_level
    xbinary = x.second_level
    
    '''Second parent'''
    y_nblock = y.n_block
    y_nfull = y.n_full
    yblocks = y.first_level
    ybinary = y.second_level
    
    '''Convolutional part crossover'''
    if x_nblock > y_nblock:
        k = math.floor(y_nblock/2)
        index = list(range(x_nblock))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(k, y_nblock):
            block = yblocks[i] #ith block
            connections = ybinary[i] #ith binary
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            yblocks[i] = xblocks[ix]
            ybinary[i] = xbinary[ix]
            xblocks[ix] = block
            xbinary[ix] = connections
    
    if y_nblock > x_nblock:
        k = math.floor(x_nblock/2)
        index = list(range(y_nblock))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(k, x_nblock):
            block = xblocks[i] #ith block
            connections = xbinary[i] #ith binary
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            xblocks[i] = yblocks[ix]
            xbinary[i] = ybinary[ix]
            yblocks[ix] = block
            ybinary[ix] = connections
    
    if x_nblock == y_nblock:
        k = math.floor(x_nblock/2)
        index = list(range(x_nblock))
        
        x_part = xblocks[k:x_nblock]
        x_binary = xbinary[k:x_nblock]
        
        '''Exchaning last half of the blocks'''
        xblocks[k:x_nblock] = yblocks[k:y_nblock]
        xbinary[k:x_nblock] = ybinary[k:y_nblock]
        yblocks[k:y_nblock] = x_part
        ybinary[k:y_nblock] = x_binary
            
    '''Fully-connected part'''
    if x_nfull > y_nfull:
        k = math.floor(y_nfull/2)
        index = list(range(x_nblock, x_nblock + x_nfull))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(y_nblock + k, y_nblock + y_nfull):
            block = yblocks[i] #ith block
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            yblocks[i] = xblocks[ix]
            xblocks[ix] = block
    
    if y_nfull > x_nfull:
        k = math.floor(x_nfull/2)
        index = list(range(y_nblock, y_nblock + y_nfull))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(x_nblock + k, x_nblock + x_nfull):
            block = xblocks[i] #ith block
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            xblocks[i] = yblocks[ix]
            yblocks[ix] = block
    
    if x_nfull == y_nfull:
        k = math.floor(x_nfull/2)
        
        x_part = xblocks[x_nblock + k:x_nblock + x_nfull]
        '''Exchaning last half of the blocks'''
        xblocks[x_nblock + k:x_nblock + x_nfull] = yblocks[y_nblock + k:y_nblock + y_nfull]
        yblocks[y_nblock + k:y_nblock + y_nfull] = x_part       
    
    
    return x, y  

def mutation(x):
    if random.uniform(0,1) < 0.5:
        '''Adding a new block'''
        if random.uniform(0,1) > 0.5:
            #Adding a fully-connected block
            layer = {'type' : 'fc',
                     'neurons' : random.choice(NEURONS)}
            
            #Choosing a random index to insert the new block
            index = list(range(x.n_block, x.n_block + x.n_full))
            ix = random.choice(index)
            
            x.first_level.insert(ix, layer)
            x.n_full += 1
        
        else:
            #Adding a Dense block
            layer = {'type' : 'conv',
                     'nfilters' : random.randint(3, 12), #Growth rate
                     'nconv' : random.randint(3, 5) #No. of Conv. layers inside block
                    }
            
            bits = []
            prev = -1
            for i in range(layer['nconv']):
                if prev < 1:
                    prev += 1
                if prev >= 1:
                    for _ in range(prev-1):
                        bits.append(random.choice([0,1]))
                    prev += 1
                    
            #Choosing a random index to insert the new block
            index = list(range(x.n_block))
            ix = random.choice(index)
            
            x.first_level.insert(ix, layer)
            x.second_level.insert(ix, bits)
            x.n_block += 1
                
    else:
        '''Changing hyperparameters in one block'''
        if random.uniform(0,1) > 0.5:
            '''Re-starting a fully-connected block'''
            index = list(range(x.n_block, x.n_block + x.n_full))
            ix = random.choice(index)
            new_layer = {'type' : 'fc',
                         'neurons' : random.choice(NEURONS)}
            #Switching fully-connected block
            x.first_level[ix] = new_layer
            
        else:
            '''Re-starting a dense block'''
            index = list(range(x.n_block))
            ix = random.choice(index)
            new_layer = {'type' : 'conv',
                         'nfilters' : random.randint(3, 12), #Growth rate
                         'nconv' : random.randint(3, 5) #No. of Conv. layers inside block
                         }
            
            #Switching convolutional block
            x.first_level[ix] = new_layer
        
            '''Modifying connections in second level'''
            i = random.choice(list(range(len(x.second_level[ix]))))
            #Flipping one bit in the second level
            if x.second_level[ix][i] == 1:
                x.second_level[ix][i] = 0
            else:
                x.second_level[ix][i] = 1
    

def selection(tournament, style):
    '''Stochastic tournament selection'''
    if style == 'max':
        if random.uniform(0,1) <= 0.8:
            p = max(tournament, key = lambda x: x[1])
        else:
            p = random.choice(tournament)
    else:
        if random.uniform(0,1) <= 0.8:
            p = min(tournament, key = lambda x: x[1])
        else:
            p = random.choice(tournament)
    
    return p
    

#random.seed(0)
e1 = Encoding(2,8,3,6,1,4)
e2 = Encoding(2,8,3,6,1,4)
#e2 = Encoding(8,8,4,4)

c1, c2 = crossover(e1,e2)
mutation(c1)