# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:12:59 2020

@author: user
"""

from torch import nn
import math
import torch

def conv_out_size(W, K):
    return W - K + 3

def pool_out_size(W, K):
    return math.floor((W - K)/2) + 1

def decoding(encoding):
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
    pad = 1
    if f_size > out_size:
        f_size = out_size - 1
    if i == 0 or i == 1:
      if layer['pool'] == 'off':
        operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = pad),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU(inplace = True)]
        in_channels = n_filters
        out_size = conv_out_size(out_size, f_size)
        o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'avg':
          p_size = layer['psize']
          if p_size > out_size:
              p_size = out_size - 1
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = pad),
                      nn.BatchNorm2d(n_filters),
                      nn.ReLU(inplace = True), 
                      nn.AvgPool2d(kernel_size = p_size, stride = 2)]
          in_channels = n_filters
          out_size = conv_out_size(out_size, f_size)
          out_size = pool_out_size(out_size, p_size)
          o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'max':
          p_size = layer['psize']
          if p_size > out_size:
              p_size = out_size - 1
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = pad),
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
        operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = pad),
                    nn.BatchNorm2d(n_filters),
                    nn.ReLU(inplace = True)]
        in_channels = n_filters
        out_size = conv_out_size(out_size, f_size)
        o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'avg':
          p_size = layer['psize']
          if p_size > out_size:
              p_size = out_size - 1
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = pad),
                      nn.BatchNorm2d(n_filters),
                      nn.ReLU(inplace = True), 
                      nn.AvgPool2d(kernel_size = p_size, stride = 2)]
          in_channels = n_filters
          out_size = conv_out_size(out_size, f_size)
          out_size = pool_out_size(out_size, p_size)
          o_sizes.append([out_size, in_channels])

      if layer['pool'] == 'max':
          p_size = layer['psize']
          if p_size > out_size:
              p_size = out_size - 1
          operation = [nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = f_size, padding = pad),
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

  ##Last layer generates the last neurons for softmax (change this for binary classification)
  classifier += [nn.Linear(n_neurons, 3)]

  return features, classifier, o_sizes

'''Networks class'''
class CNN(nn.Module):
  def __init__(self, encoding, features, classifier, sizes, init_weights = True):
    super(CNN, self).__init__()
    extraction = []
    for layer in features:
      extraction += layer
    self.extraction = nn.Sequential(*extraction)
    self.classifier = nn.Sequential(*classifier)
    self.features = features
    self.second_level = encoding.second_level
    self.sizes = sizes
    
  def forward(self, x):
    '''Feature extraction'''
    prev = -1
    pos = 0
    outputs = {}
    features = self.features
    #print(x.shape)
    for i in range(len(features)):
      #print('Layer: ', i)
      if i == 0 or i == 1:
        x = nn.Sequential(*features[i])(x)
        outputs[i] = x
        #print(x.shape)
      
      else:
        connections = self.second_level[pos:pos+prev]
        for c in range(len(connections)):
          if connections[c] == 1:
            skip_size = self.sizes[c][0] #Size comming from previous layer
            req_size = x.shape[2] #Current feature map size
            #print('X: ',x.shape)
            if skip_size > req_size:
              psize = skip_size - req_size + 1 
              pool = nn.MaxPool2d(kernel_size = psize, stride = 1) #Applying pooling to adjust sizes
              x2 = pool(outputs[c])
            if skip_size == req_size:
              x2 = outputs[c]
            if req_size == skip_size + 1:
              pool = nn.MaxPool2d(kernel_size = 2, stride = 1, padding = (1,1))
              x2 = pool(outputs[c])
            if req_size == skip_size + 2:
              pad = int((req_size - skip_size)/2)
              padding = nn.ZeroPad2d(pad)
              x2 = padding(outputs[c])
            #print('X2: ',x2.shape)
            x = torch.cat((x, x2), axis = 1)
          
        x = nn.Sequential(*features[i])(x)
        #print('Out size: ', x.shape)
        outputs[i] = x
        pos += prev
      
      prev += 1
    
    #print('Classification size: ', x.shape)
    x = torch.flatten(x,1)
    '''Classification'''
    '''for l in self.classifier:
      x = l(x)'''
    x = self.classifier(x)
    #print(x.shape)
    return nn.functional.log_softmax(x, dim=1)