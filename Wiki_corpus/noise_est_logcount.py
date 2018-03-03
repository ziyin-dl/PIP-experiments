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
log_count = np.log(1 + Nij)

Nij_test = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    Nij_test[i,j] += cooccur_test[i][j]

log_count_test = np.log(1 + Nij_test)
diff = log_count - log_count_test
print(np.mean(diff))
print(0.5 * np.std(diff))

