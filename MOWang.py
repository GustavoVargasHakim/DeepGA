# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:31:09 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:37:31 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:43:55 2020

@author: user
"""

from WangOperators import *
from WangEncoding import *
from WangDecoding import *
from DataReader import *
from Training import *
from DistributedTraining import *
import numpy as np
from torch import optim
import pandas as pd
import timeit
import torch
from torch import nn
from multiprocessing import Process, Manager
import pickle

def dominates(p, q):
    if p[1] >= q[1] and p[2] <= q[2]:
        if p[1] > q[1] or p[2] < q[2]:
            dom = True
        else:
            dom = False
    else:
        dom = False
            
    return dom


def dominance_sorting(Pop):
    R = Pop[:]
    F = []
    while len(R) != 0:
        p = R[0]
        Fi = [p]
        for q in [Q for Q in R if Q not in Fi]:
            if len(Fi) == 1:
                p = Fi[0]
                if dominates(q, p):
                    p = q
                    Fi[0] = q
                elif not dominates(p, q) and not dominates(q,p):
                    Fi.append(q)
            else:
                n = 0
                Fic = Fi[:]
                for j in range(len(Fic)):
                    f = Fic[j]
                    if dominates(q, f):
                        if q not in Fi:
                            Fi[j] = q
                        else:
                            Fi.remove(f)
                    elif not dominates(f, q) and not dominates(q,f):
                        n += 1
                if n == len(Fi):
                    Fi.append(q)
        F += Fi
        for f in Fi:
            R.remove(f)
    
    return F
                

#Random seed
#random.seed(1)
#torch.manual_seed(1)

#Loading data
train_dl, test_dl = loading_data()

'''Defining CNN hyperparameters'''
#Defining loss function
loss_func = nn.NLLLoss(reduction = "sum")

#Defining learning rate
lr = 1e-4

#Maximun and minimum numbers of layers to initialize networks
min_conv = 2
max_conv = 5
min_gr = 3
max_gr = 12
min_full = 1
max_full = 4

'''Genetic Algorithm Parameters'''
cr = 0.7 #Crossover rate
mr = 0.5 #Mutation rate
N = 20 #Population size
T = 20 #Number of generations
t_size = 5 #tournament size
w = 0.3 #penalization weight
max_params = 2e6
num_epochs = 10

#Reading GPU
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")

'''Initialize population'''
print('Initialize population')
start = timeit.default_timer()
pop = []
bestAcc = []
bestF = []
bestParams = []
manager = Manager()
while len(pop) < N:
    acc_list = manager.list()
    
    #Creating genomes (genetic encoding)
    e1 = Encoding(min_conv, max_conv, min_gr, max_gr, min_full, max_full) 
    e2 = Encoding(min_conv, max_conv, min_gr, max_gr, min_full, max_full)
    
    #Decoding the networks
    network1 = decoding(e1)
    network2 = decoding(e2)
    
    #Creating the CNNs 
    cnn1 = CNN(e1, network1[0], network1[1], network1[2])
    cnn2 = CNN(e2, network2[0], network2[1], network2[2])
       
    #Evaluate individuals
    training1 = Process(target = training, args = ('1', device1, cnn1, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
    
    training2 = Process(target = training, args = ('2', device2, cnn2, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
    
    training1.start()
    training2.start()
    training1.join()
    training2.join()
    
    if acc_list[0][0] == '1':
        pop.append([e1, acc_list[0][2], acc_list[0][3]])
        pop.append([e2, acc_list[1][2], acc_list[1][3]])
    else:
        pop.append([e2, acc_list[0][2], acc_list[0][3]])
        pop.append([e1, acc_list[1][2], acc_list[1][3]])

'''Genetic Algorithm'''
for t in range(T):
    print('Generation: ', t)
    
    #Parents Selection
    parents = []
    while len(parents) < int(N/2):
        #Tournament Selection
        tournament = random.sample(pop, t_size)
        p1 = selection(tournament, 'max')
        tournament = random.sample(pop, t_size)
        p2 = selection(tournament, 'max')
        while p1 == p2:
            tournament = random.sample(pop, t_size)
            p2 = selection(tournament, 'max')  
        
        parents.append(p1)
        parents.append(p2)
    
    #Reproduction
    offspring = []
    while len(offspring) < int(N/2):
        par = random.sample(parents, 2)
        #Crossover + Mutation
        if cr >= random.uniform(0,1): #Crossover
            p1 = par[0][0]
            p2 = par[1][0]
            c1, c2 = crossover(p1, p2)
            
            #Mutation
            if mr >= random.uniform(0,1):
                mutation(c1)
            
            if mr >= random.uniform(0,1):
                mutation(c2)
            
            #Evaluate offspring
            acc_list = manager.list()
            
            #Decoding the network
            network1 = decoding(c1)
            network2 = decoding(c2)
    
            #Creating the CNN 
            cnn1 = CNN(c1, network1[0], network1[1], network1[2])
            cnn2 = CNN(c2, network2[0], network2[1], network2[2])
            
            #Evaluate individuals
            training1 = Process(target = training, args = ('1', device1, cnn1, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
    
            training2 = Process(target = training, args = ('2', device2, cnn2, num_epochs, loss_func, 
                                                  train_dl, test_dl, lr, w, max_params, acc_list))
            
            training1.start()
            training2.start()
            training1.join()
            training2.join()
            
            if acc_list[0][0] == '1':
                offspring.append([c1, acc_list[0][2], acc_list[0][3]])
                offspring.append([c2, acc_list[1][2], acc_list[1][3]])
            else:
                offspring.append([c2, acc_list[0][2], acc_list[0][3]])
                offspring.append([c1, acc_list[1][2], acc_list[1][3]])
       
    #Replacement with elitism
    pop = pop + offspring
    pop = dominance_sorting(pop)
    pop = pop[:N]
    
    leader = max(pop, key = lambda x: x[1])
    #bestAcc.append(leader[2])
    #bestF.append(leader[1])
    #bestParams.append(leader[3])
    
        
    print('Best accuracy: ', leader[1])
    print('Best No. of Params: ', leader[2])
    print('No. of Conv. Layers: ', leader[0].n_block)
    print('No. of FC Layers: ', leader[0].n_full)
    print('--------------------------------------------')

stop = timeit.default_timer()
execution_time = (stop-start)/3600
print("Execution time: ", execution_time)
accuracy = []
parameters = []
for p in pop:
    accuracy.append(p[1])
    parameters.append(p[2])

results = pd.DataFrame(list(zip(accuracy, parameters)), columns = ['Accuracy', 'No. Params'])
final_networks = []
final_connections = []
objects = []
for member in pop:
    p = member[0]
    objects.append(p)
    n_block = p.n_block
    n_full = p.n_full
    connections = '( '
    description = 'The network has ' + str(n_block) + ' dense blocks ' + 'with: '
    for i in range(n_block):
        nfilters = str(p.first_level[i]['nfilters'])
        n_conv = str(p.first_level[i]['nconv'])
        conn = p.second_level[i]
        
        layer = '(' + nfilters + ', ' + n_conv  + ') '
        description += layer
        
        
        block = '[ '
        for bit in p.second_level[i]:
            if bit == 1:
                block += 'one - '  
            if bit == 0:
                block += 'zero - '
        block += ']'
        connections += block
    final_connections.append(connections)
    description += 'and '
    description += str(n_full)
    description += ' '
    description += 'fully-connected layers with: '
    for i in range(n_block, n_block+n_full):
        neurons = str(p.first_level[i]['neurons'])
        layer = '(' + neurons + ')'
        description += layer
    description += ' neurons'
    final_networks.append(description)

     
final_population = pd.DataFrame(list(zip(final_networks, final_connections)), columns = ['Network Architecture', 'Connections'])

'''Saving Results as CSV'''
final_population.to_csv('/home/proy_ext_adolfo.vargas/DeepGA/final_population2.csv', index = False)
final_population.to_csv('final_population2.csv', index = False)
results.to_csv('/home/proy_ext_adolfo.vargas/DeepGA/results2.csv', index = False)
results.to_csv('results2.csv', index = False)      


#Saving objects
with open('/home/proy_ext_adolfo.vargas/cnns2.pkl', 'wb') as output:
    pickle.dump(objects, output, pickle.HIGHEST_PROTOCOL)
    output.close()

with open('cnns2.pkl', 'wb') as output:
    pickle.dump(objects, output, pickle.HIGHEST_PROTOCOL)
    output.close()
    

