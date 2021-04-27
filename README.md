# DeepGA
DeepGA is a novel Neuroevolution framework to evolve Convolutional Neural Networks, using a Genetic Algorithm.  

* **Encoding**: the neural encoding in DeepGA is a low-modular, flexible hybrid representation based on convolutional blocks (convolution + optional pooling) and fully connected blocks (linear layers), and binary strings that represent the dense connectivity patterns in the convolutional part of the CNN (as seen in the image below). This encoding is unbiased towards larger CNNs, such as the *Wang encoding*, which is used as the comparative standpoint \(https://link.springer.com/chapter/10.1007/978-3-030-29894-4_52). Also, a series of evolutionary operators (selection, crossover, and mutation) have been designed to deal with this encoding.


* **Fitness**: the single objective version of DeepGA utilizes a linear weighted fitness functions consisting on the accuracy and the number of parameters of the CNN. This aids at searching for CNN architectures with a high classification accuracy and a low number of trainable parameters (weights, biases), based on the user requirements. The Multi-Objective version of DeepGA, on the other hand, takes both the accuracy (or classification error) and the number of parameters of the CNNs directly as two objective functions.
 
<img src="Images/NewEncoding.png" width="581" height="364">

## Images

The chest X-ray images have been obtained from (https://github.com/ari-dasci/OD-covidg, https://github.com/ieee8023/covid-chestxray-dataset, and https://github.com/agchung/Actualmed-COVID-chestxray-dataset). The available images are categorized as COVID-19, viral/bacterial pneumonia, and healthy. A prepropcessing has already been applied to the entire image set in order to augment their quality. Please see https://ieeexplore.ieee.org/document/9090149 for more details.

## How to use DeepGA?

All the files in DeepGA are based on Python 3.7, using PyTorch as the Deep Learning framework. To use the single objective version, **DeepGA.py** must be run. Notice that the configuration of this program is made for a dual-GPU training; two CNNs are trained at the same time during the GA. The program **DistributedTraining.py** contains this configuration (but it should not be modified). It is highly encouraged to use at least two GPU to speed up the Neuroevolution process. 

The Multi-Objective version of DeepGA, **MODeepGA.py**, contains the necessary adjustments for DeepGA to work with two objective functions. Notice that, when the Wang encoding is utilized, the programs for single- and multi-objective Neuroevolution are **WangDeepGA.py** and **MOWang.py**, respectively. In both cases, the configuration needs little to no changes. 

The user-available parameters of DeepGA are:

```
lr = 1e-4 #Learning rate

'''Initialization'''
min_conv = 2 #Minimum number of convolutional blocks
max_conv = 5 #Maximum number of convolutional blocks
min_full = 1 #Minimum number of fully-connected blocks
max_full = 4 #Maximum number of fully-connected blocks

'''Genetic Algorithm parameters'''
cr = 0.7 #Crossover rate
mr = 0.5 #Mutation rate
N = 20 #Population size
T = 50 #Number of generations
t_size = 5 #tournament size
w = 0.3 #penalization weight (for single-objective DeepGA only)
max_params = 2e6
num_epochs = 10
```

## How to properly cite DeepGA?

DeepGA is the source code of the paper "Hybrid Encodings for Neuroevolution of Convolutional Neural Networks: A Case Study" published at the Workshop Neuroevolution at Work from the ACM Genetic and Evolutionary Computation Conference (GECCO'21). A citation of this paper when using the source code is highly appreciated:

```
@Inproceedings{DeepGA2021,
  author =  "Gustavo-Adolfo Vargas-Hákim and Efrén Mezura-Montes and Héctor-Gabriel Acosta-Mesa",
  title =        "Hybrid Encodings for Neuroevolution of Convolutional Neural Networks: A Case Study",
  booktitle =    "Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion",
  year =         "2021",
  publisher =    "Association for Computing Machinery",
  pages =        "",
  doi = 	 "",
}
```
