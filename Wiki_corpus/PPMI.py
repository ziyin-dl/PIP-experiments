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

vocabulary_size = 10245
save_path = './dat_enriched_{}/'.format(vocabulary_size)
try:
  with open(save_path + 'dictionary.pkl', 'r') as f:
    dictionary = pickle.load(f)
  with open(save_path + 'reverse_dictionary.pkl', 'r') as f:
    reverse_dictionary = pickle.load(f)

except Exception, e:
  print("error loading files: {}".format(e))

try:
  with open(save_path + 'cooccur.pkl', 'r') as f:
    cooccur = pickle.load(f)
  with open(save_path + 'cooccur_matrix.pkl', 'r') as f:
    cooccur_matrix = pickle.load(f)

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
# SPPMI = np.maximum(PMI - np.log(4), 0.0)

U, D, V = np.linalg.svd(PPMI)
plt.plot(D)
plt.savefig('sv_PPMI_wiki_top{}.pdf'.format(vocabulary_size))
plt.close()

with open('sv_PPMI_wiki.pkl', 'w') as f:
  pickle.dump(D, f)

dir_name = 'test_results'
sp.check_output('mkdir -p {}'.format(dir_name), shell=True)

for idx, alpha in enumerate(args.alpha):
  sim_tests = {}
  for test_file in test_set_files:
    sim_tests[test_file] = []
  sim_best_dim = {}
  for test_file in test_set_files:
    sim_best_dim[test_file] = (0, 0)
  plt.figure(idx)
  alpha = float(alpha)
  embeddings_U_tot = U * np.power(D, alpha)

  for keep_dim in range(1, 1 + len(D)):
    print(keep_dim)
    embeddings_U = embeddings_U_tot[:,:keep_dim]
    for test_file, test_rows in zip(test_set_files, test_rows_list):
      score1, score2 = test(test_rows, dictionary, embeddings_U)
      corr = np.corrcoef(score1, score2)[0,1]
      sim_tests[test_file].append(corr)
      if corr > sim_best_dim[test_file][1]:
        sim_best_dim[test_file] = (keep_dim, corr)
  with open('{}/test_result_{}.pkl'.format(dir_name, alpha), 'w') as f:
    pickle.dump(sim_tests, f)
  for k, v in sim_tests.iteritems():
    plt.plot(v, label=k.split('.')[0])
    plt.legend(loc='lower right')
    plt.title('Similarity Task Performance vs Embedding Size')
    plt.ylabel('correlation with hunam labels')
    plt.xlabel('dimensions')
  plt.savefig('{}/scores_{}.pdf'.format(dir_name, alpha))
  plt.close()
