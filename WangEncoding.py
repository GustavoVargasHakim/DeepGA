# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:57:16 2020

@author: user
"""
import random

'''Hyperparameters configuration'''
#Fully connected layers
NEURONS = [4,8,16,32,64,128]

class Encoding:
    def __init__(self, minC, maxC, mink, maxk, minF, maxF):
        self.n_block = random.randint(minC, maxC)
        self.n_full = random.randint(minF, maxF)
        
        
        '''First level encoding'''
        self.first_level = []
        
        '''Second level encoding'''
        self.second_level = []
        
        #Feature extraction part
        for i in range(self.n_block):
            layer = {'type' : 'conv',
                     'nfilters' : random.randint(mink, maxk), #Growth rate
                     'nconv' : random.randint(3, 5) #No. of Conv. layers inside block
                    }
            self.first_level.append(layer)
            bits = []
            prev = -1
            for i in range(layer['nconv']):
                if prev < 1:
                    prev += 1
                if prev >= 1:
                    for _ in range(prev-1):
                        bits.append(random.choice([0,1]))
                    prev += 1
            self.second_level.append(bits)
        
        #Fully connected part
        for i in range(self.n_full):
            layer = {'type' : 'fc',
                     'neurons' : random.choice(NEURONS)}
            
            self.first_level.append(layer)
        

                
