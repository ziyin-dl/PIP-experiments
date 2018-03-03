from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import cPickle as pickle
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable
from scipy.sparse import csr_matrix
import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

vocabulary_size = 10000
save_path = './dat_splitted_{}/'.format(vocabulary_size)
try:
  with open(save_path + 'dictionary_0.pkl', 'r') as f:
    dictionary = pickle.load(f)
  with open(save_path + 'reverse_dictionary_0.pkl', 'r') as f:
    reverse_dictionary = pickle.load(f)
  with open(save_path + 'dictionary_1.pkl', 'r') as f:
    dictionary_test = pickle.load(f)
  with open(save_path + 'reverse_dictionary_1.pkl', 'r') as f:
    reverse_dictionary_test = pickle.load(f)

except Exception, e:
  print("error loading files: {}".format(e))

try:
  with open(save_path + 'cooccur_0.pkl', 'r') as f:
    cooccur = pickle.load(f)
  with open(save_path + 'cooccur_1.pkl', 'r') as f:
    cooccur_test = pickle.load(f)
  with open(save_path + 'cooccur_matrix_0.pkl', 'r') as f:
    cooccur_matrix = pickle.load(f)
  cooccur_test_matrix = np.zeros([vocabulary_size, vocabulary_size])
  with open(save_path + 'cooccur_matrix_1.pkl', 'r') as f:
    cooccur_test_matrix = pickle.load(f)

except Exception, e:
  print("error loading files: {}".format(e))

Nij = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    Nij[i,j] += cooccur[i][j]

Ni = np.sum(Nij, axis = 1)
tot = np.sum(Ni)
Pij = Nij / tot 
Pi = Ni / np.sum(Ni)
print(np.sum(Pij))
print(np.sum(Pi))


PMI = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    if Pi[i] * Pi[j] > 0 and Pij[i,j] > 0:
      PMI[i,j] = np.log(Pij[i,j] / (Pi[i] * Pi[j]))

PPMI = np.maximum(PMI, 0.0)
SPPMI = np.maximum(PMI - np.log(4), 0.0)

Nij_test = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    Nij_test[i,j] += cooccur_test[i][j]
Ni_test = np.sum(Nij, axis = 1)
tot_test = np.sum(Ni)
Pij_test = Nij_test / tot 
Pi_test = Ni_test / np.sum(Ni_test)
print(np.sum(Pij_test))
print(np.sum(Pi_test))


PMI_test = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    if Pi_test[i] * Pi_test[j] > 0 and Pij_test[i,j] > 0:
      PMI_test[i,j] = np.log(Pij_test[i,j] / (Pi_test[i] * Pi_test[j]))

PPMI_test = np.maximum(PMI_test, 0.0)
SPPMI_test = np.maximum(PMI_test - np.log(4), 0.0)

diff = PPMI - PPMI_test
print(np.mean(diff))
print(0.5 * np.std(diff))

