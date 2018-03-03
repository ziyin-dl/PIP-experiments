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
from six.moves import xrange  # pylint: disable=redefined-builtin
import bisect
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds as sparse_svd
from numpy.linalg import svd as svd
from numpy.linalg import norm as norm
is_sparse = False

def test(embeddings_U, scores):
  score1, score2 = [], []
  for k, v in scores.iteritems():      
    score1.append(v)
    row1, row2 = k[0], k[1]
    score2.append(embeddings_U[row1,:].dot(embeddings_U[row2,:].T) / (norm(embeddings_U[row1,:], 2) * norm(embeddings_U[row2,:], 2)))
  return score1, score2

vocabulary_size = 2179 

save_path = './dat_{}/'.format(vocabulary_size)
with open(save_path + 'dictionary.pkl', 'r') as f:
  dictionary = pickle.load(f)
with open(save_path + 'reverse_dictionary.pkl', 'r') as f:
  reverse_dictionary = pickle.load(f)

with open(save_path + 'tf_idf.pkl', 'r') as f:
  tf_idf = pickle.load(f)

with open(save_path + 'scores.pkl', 'r') as f:
  scores = pickle.load(f)

if is_sparse:
  tf_idf = csc_matrix(tf_idf)

tf_idf_top = []

tf_idf = tf_idf.T
print('Doing svd...')
if is_sparse:
  U, D, V = sparse_svd(tf_idf, min(tf_idf.shape)) #, k = min(PPMI.shape) - 1)
  U = U.T
else:
  U, D, V = svd(tf_idf, full_matrices = False)
print('finished doing svd.')
embeddings_U_tot = U * np.sqrt(D)

tot_l = 1
tot_u = min(vocabulary_size, D.shape[0])

for i in range(tot_l, tot_u):
  print(i)
  keep_dim = i
  
  keep_dims = np.arange(keep_dim)
  embeddings_U = embeddings_U_tot[:, keep_dims]

  score1, score2 = test(embeddings_U, scores)
  tf_idf_top.append(np.corrcoef(score1, score2)[0,1])

plt.title('TF-iDF Document Similarity w.r.t. Dimensions')
plt.plot(tf_idf_top, 'g', label='TF-iDF')
plt.xlabel('Dimensions')
plt.ylabel('Correlation with Human Labels')
plt.legend(loc='upper right')
plt.savefig('tf_idf_SYM_cos.pdf')
plt.close()
with open('tfidf_sim_accuracy.pkl', 'w') as f:
  pickle.dump(tf_idf_top, f)


exit()

