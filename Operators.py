# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:11:23 2020

@author: user
"""

import random
import math
from copy import deepcopy
from EncodingClass import *

def crossover(x, y):
    x = deepcopy(x)
    y = deepcopy(y)
    
    '''First parent'''
    x_nconv = x.n_conv
    x_nfull = x.n_full
    xblocks = x.first_level
    xbinary = x.second_level
    
    '''Second parent'''
    y_nconv = y.n_conv
    y_nfull = y.n_full
    yblocks = y.first_level
    ybinary = y.second_level
    
    '''Convolutional part crossover'''
    if x_nconv > y_nconv:
        k = math.floor(y_nconv/2)
        index = list(range(x_nconv))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(k, y_nconv):
            block = yblocks[i] #ith block
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            yblocks[i] = xblocks[ix]
            xblocks[ix] = block
    
    if y_nconv > x_nconv:
        k = math.floor(x_nconv/2)
        index = list(range(y_nconv))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(k, x_nconv):
            block = xblocks[i] #ith block
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            xblocks[i] = yblocks[ix]
            yblocks[ix] = block
    
    if x_nconv == y_nconv:
        k = math.floor(x_nconv/2)
        index = list(range(x_nconv))
        
        x_part = xblocks[k:x_nconv]
        
        '''Exchaning last half of the blocks'''
        xblocks[k:x_nconv] = yblocks[k:y_nconv]
        yblocks[k:y_nconv] = x_part
            
    '''Fully-connected part'''
    if x_nfull > y_nfull:
        k = math.floor(y_nfull/2)
        index = list(range(x_nconv, x_nconv + x_nfull))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(y_nconv + k, y_nconv + y_nfull):
            block = yblocks[i] #ith block
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            yblocks[i] = xblocks[ix]
            xblocks[ix] = block
    
    if y_nfull > x_nfull:
        k = math.floor(x_nfull/2)
        index = list(range(y_nconv, y_nconv + y_nfull))
        
        '''Exchanging the last k blocks of the smaller parent'''
        for i in range(x_nconv + k, x_nconv + x_nfull):
            block = xblocks[i] #ith block
            ix = random.choice(index) #Selecting random index from larger parent
            index.remove(ix)
            
            #Exchange of blocks
            xblocks[i] = yblocks[ix]
            yblocks[ix] = block
    
    if x_nfull == y_nfull:
        k = math.floor(x_nfull/2)
        
        x_part = xblocks[x_nconv + k:x_nconv + x_nfull]
        '''Exchaning last half of the blocks'''
        xblocks[x_nconv + k:x_nconv + x_nfull] = yblocks[y_nconv + k:y_nconv + y_nfull]
        yblocks[y_nconv + k:y_nconv + y_nfull] = x_part       
    
    '''Second level'''
    if len(xbinary) > len(ybinary):
        if len(ybinary) > 1 :
            k = random.choice(list(range(1, len(ybinary))))
            partition = ybinary[k:]
            nbits = len(partition)
        
            if random.uniform(0,1) >= 0.5:
                ybinary[k:] = xbinary[len(xbinary) - nbits:len(xbinary)]
                xbinary[len(xbinary) - nbits:len(xbinary)] = partition
            else:
                ybinary[k:] = xbinary[:nbits]
                xbinary[:nbits] = partition
    
    if len(ybinary) > len(xbinary):
        if len(xbinary) > 1 :
            k = random.choice(list(range(len(xbinary))))
            partition = xbinary[k:]
            nbits = len(partition)
            
            if random.uniform(0,1) >= 0.5:
                xbinary[k:] = ybinary[len(ybinary) - nbits:len(ybinary)]
                ybinary[len(ybinary) - nbits:len(ybinary)] = partition
            else:
                xbinary[k:] = ybinary[:nbits]
                ybinary[:nbits] = partition
        
    if len(xbinary) == len(ybinary):
        if len(xbinary) > 1 :
            k = random.choice(list(range(len(xbinary))))
            partition = xbinary[k:]
        
            xbinary[k:] = ybinary[k:]
            ybinary[k:] = partition
    
    return x, y  

def mutation(x):
    if random.uniform(0,1) < 0.5:
        '''Adding a new block'''
        if random.uniform(0,1) > 0.5:
            #Adding a fully-connected block
            layer = {'type' : 'fc',
                     'neurons' : random.choice(NEURONS)}
            
            #Choosing a random index to insert the new block
            index = list(range(x.n_conv, x.n_conv + x.n_full))
            ix = random.choice(index)
            
            x.first_level.insert(ix, layer)
            x.n_full += 1
        
        else:
            #Adding a convolutional block
            layer = {'type' : 'conv',
                     'nfilters' : random.choice(NFILTERS),
                     'fsize' : random.choice(FSIZES),
                     'pool' : random.choice(['max', 'avg', 'off']),
                     'psize' : random.choice(PSIZES)
                    }
            #Choosing a random index to insert the new block
            index = list(range(x.n_conv))
            ix = random.choice(index)
            
            x.first_level.insert(ix, layer)
            x.n_conv += 1
            
            if ix > 1:
                new_bits = []
                for i in range(ix - 1):
                    new_bits.append(random.choice([0,1]))
                pos = int(0.5*(ix**2) - 1.5*(ix) + 1)
                start = pos + len(new_bits)
                for bit in new_bits:
                    x.second_level.insert(pos, bit)
                    pos += 1
                
                rest = x.n_conv - ix - 1
                add = ix
                for j in range(rest):
                    x.second_level.insert(start+add-1, random.choice([0,1]))
                    start += add
                    ix += 1
            
            if ix == 0 or ix == 1:
                if x.n_conv - 1 == 2:
                    x.second_level.append(random.choice([0,1]))
                else:
                    add = 0
                    for i in range(2, x.n_conv):
                        pos = int(0.5*(ix**2) - 1.5*(ix) + 1) + add
                        x.second_level.insert(pos, random.choice([0,1]))
                        add += 1
                
    else:
        '''Changing hyperparameters in one block'''
        if random.uniform(0,1) > 0.5:
            '''Re-starting a fully-connected block'''
            index = list(range(x.n_conv, x.n_conv + x.n_full))
            ix = random.choice(index)
            new_layer = {'type' : 'fc',
                         'neurons' : random.choice(NEURONS)}
            #Switching fully-connected block
            x.first_level[ix] = new_layer
            
        else:
            '''Re-starting a convolutional block'''
            index = list(range(x.n_conv))
            ix = random.choice(index)
            new_layer = {'type' : 'conv',
                     'nfilters' : random.choice(NFILTERS),
                     'fsize' : random.choice(FSIZES),
                     'pool' : random.choice(['max', 'avg', 'off']),
                     'psize' : random.choice(PSIZES)
                    }
            
            #Switching convolutional block
            x.first_level[ix] = new_layer
        
        '''Modifying connections in second level'''
        if len(x.second_level) > 0:
            k = random.choice(list(range(len(x.second_level))))
            #Flipping one bit in the second level
            if x.second_level[k] == 1:
                x.second_level[k] = 0
            else:
                x.second_level[k] = 1
    

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
    

random.seed(0)
e1 = Encoding(2,8,1,4)
e2 = Encoding(2,8,1,4)
#e2 = Encoding(8,8,4,4)

c1, c2 = crossover(e1,e2)