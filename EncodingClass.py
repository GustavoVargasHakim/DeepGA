# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:57:16 2020

@author: user
"""
import random

'''Hyperparameters configuration'''
#Convolutional layers
FSIZES = [2,3,4,5,6,7,8]
NFILTERS = [2,4,8,16,32]

#Pooling layers
PSIZES = [2,3,4,5]
PTYPE = ['max', 'avg']

#Fully connected layers
NEURONS = [4,8,16,32,64,128]

class Encoding:
    def __init__(self, minC, maxC, minF, maxF):
        self.n_conv = random.randint(minC, maxC)
        self.n_full = random.randint(minF, maxF)
        
        
        '''First level encoding'''
        self.first_level = []
        
        #Feature extraction part
        for i in range(self.n_conv):
            layer = {'type' : 'conv',
                     'nfilters' : random.choice(NFILTERS),
                     'fsize' : random.choice(FSIZES),
                     'pool' : random.choice(['max', 'avg', 'off']),
                     'psize' : random.choice(PSIZES)
                    }
            self.first_level.append(layer)
        
        #Fully connected part
        for i in range(self.n_full):
            layer = {'type' : 'fc',
                     'neurons' : random.choice(NEURONS)}
            
            self.first_level.append(layer)
        
        
        '''Second level encoding'''
        self.second_level = []
        prev = -1
        for i in range(self.n_conv):
            if prev < 1:
                prev += 1
            if prev >= 1:
                for _ in range(prev-1):
                    self.second_level.append(random.choice([0,1]))
                prev += 1
                
'''def decoding(encoding):
  n_conv = encoding.n_conv
  n_full = encoding.n_full
  first_level = encoding.first_level
  second_level = encoding.second_level

  features = []
  classifier = []
  in_channels = 1
  out_size = 256
  prev = -1
  pos = 0
  o_sizes = []
  for i in range(n_conv):
    layer = first_level[i]
    n_filters = layer['nfilters']
    f_size = layer['fsize']
    block = []
    if i == 0 or i == 1:
      if layer['pool'] == 'off':
        operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = 1),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU(inplace = True)]
        in_channels = n_filters
        out_size = conv_out_size(out_size, f_size)
        o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'avg':
          p_size = layer['psize']
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = 1),
                      nn.BatchNorm2d(n_filters),
                      nn.ReLU(inplace = True), 
                      nn.AvgPool2d(kernel_size = p_size, stride = 2)]
          in_channels = n_filters
          out_size = conv_out_size(out_size, f_size)
          out_size = pool_out_size(out_size, p_size)
          o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'max':
          p_size = layer['psize']
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = 1),
                      nn.BatchNorm2d(n_filters),
                      nn.ReLU(inplace = True), 
                      nn.MaxPool2d(kernel_size = p_size, stride = 2)]
          in_channels = n_filters
          out_size = conv_out_size(out_size, f_size)
          out_size = pool_out_size(out_size, p_size)
          o_sizes.append([out_size, in_channels])
    else:
      connections = second_level[pos:pos+prev]
      for c in range(len(connections)):
        if connections[c] == 1:
          in_channels += o_sizes[c][1]  

      if layer['pool'] == 'off':
        operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = 1),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU(inplace = True)]
        in_channels = n_filters
        out_size = conv_out_size(out_size, f_size)
        o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'avg':
          p_size = layer['psize']
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = 1),
                      nn.BatchNorm2d(n_filters),
                      nn.ReLU(inplace = True), 
                      nn.AvgPool2d(kernel_size = p_size, stride = 2)]
          in_channels = n_filters
          out_size = conv_out_size(out_size, f_size)
          out_size = pool_out_size(out_size, p_size)
          o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'max':
          p_size = layer['psize']
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = 1),
                      nn.BatchNorm2d(n_filters),
                      nn.ReLU(inplace = True), 
                      nn.MaxPool2d(kernel_size = p_size, stride = 2)]
          in_channels = n_filters
          out_size = conv_out_size(out_size, f_size)
          out_size = pool_out_size(out_size, p_size)
          o_sizes.append([out_size, in_channels])

      pos += prev
    prev += 1
      
    features.append(operation)
  in_size = out_size*out_size*in_channels
  for i in range(n_conv,(n_conv + n_full)):
    layer = first_level[i]
    n_neurons = layer['neurons']
    classifier += [nn.Linear(in_size, n_neurons)]
    classifier += [nn.ReLU(inplace = True)]
    in_size = n_neurons

  return features, classifier, o_sizes'''
       
            
                
            
        