from torch import nn
import math
import torch

def conv_out_size(W, K):
    return W - K + 1

def pool_out_size(W, K):
    return math.floor((W - K)/2) + 1

def decoding(encoding):
  n_block = encoding.n_block
  n_full = encoding.n_full
  first_level = encoding.first_level
  second_level = encoding.second_level

  '''Components'''
  DenseBlock = []
  links = []
  classifier = []
  input_channels = 1
  out_size = 256
  for i in range(n_block):
    block = first_level[i]
    connections = second_level[i]
    n_conv = block['nconv']
    g_rate = block['nfilters']
    prev = -1
    pos = 0
    layers = []
    for j in range(n_conv):
      if j == 0 or j == 1:
        layers.append([nn.Conv2d(in_channels = input_channels, out_channels = g_rate, kernel_size = 3, padding = 1, stride = 1),
                      nn.BatchNorm2d(g_rate),
                      nn.ReLU(inplace = True)])
      else:
        conn = connections[pos:pos+prev]
        new_inputs = 0
        for c in range(len(connections[pos:pos+prev])):
          if conn[c] == 1:
            new_inputs += g_rate
        
        layers.append([nn.Conv2d(in_channels = (input_channels+new_inputs), out_channels = g_rate, kernel_size = 3, padding = 1, stride = 1),
                      nn.BatchNorm2d(g_rate),
                      nn.ReLU(inplace = True)])
        pos += prev
      prev += 1
      input_channels = g_rate
    DenseBlock.append(layers)
    links.append([nn.Conv2d(in_channels = input_channels, out_channels = int(g_rate/2), kernel_size = 3, padding = 0, stride = 1),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)])
    input_channels = int(g_rate/2) 
    out_size = conv_out_size(out_size, 3)
    out_size = pool_out_size(out_size, 2)


  in_size = out_size*out_size*input_channels  
  classifier = []
  for i in range(n_block,n_block+n_full):
    block = first_level[i]
    n_neurons = block['neurons']
    classifier += [nn.Linear(in_size, n_neurons)]
    in_size = n_neurons
  
  classifier += [nn.Linear(n_neurons, 3)]
  
  return DenseBlock, links, classifier

'''Networks class'''
class CNN(nn.Module):
  def __init__(self, e, denseBlocks, links, classifier, init_weights = True):
    super(CNN, self).__init__()
    self.extraction = nn.ModuleList()
    for block in denseBlocks:
      blocks = nn.ModuleList()
      for layer in block:
        blocks.append(nn.Sequential(*layer))
      self.extraction.append(blocks)
    self.links = nn.ModuleList()
    for link in links:
      self.links.append(nn.Sequential(*link))
    self.classifier = nn.Sequential(*classifier)
    self.denseBlocks = denseBlocks
    self.connections = e.second_level
    self.first_level = e.first_level
    self.nblocks = e.n_block
    
    
  def forward(self, x):
    '''Feature extraction'''
    for i in range(self.nblocks):
        block = self.extraction[i]
        connections = self.connections[i]
        link = self.links[i]
        prev = -1
        pos = 0
        outputs = []
        for j in range(len(block)):
          if j == 0 or j == 1:
            x = block[j](x)
            outputs.append(x)
          else:
            conn = connections[pos:pos+prev]
            for c in range(len(conn)):
              if conn[c] == 1:
                x2 = outputs[c]
                x = torch.cat((x, x2), axis = 1)
            x = block[j](x)
            outputs.append(x)
            pos += prev
          prev += 1
        x = nn.Sequential(*link)(x)

    x = torch.flatten(x,1)
    '''Classification'''
    x = self.classifier(x)
    return nn.functional.log_softmax(x, dim=1)