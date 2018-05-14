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
import subprocess as sp
import argparse

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'
parser = argparse.ArgumentParser(description='experiment with the effect of signal/noise for the invariant spaces.')


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.show()
  plt.savefig(filename)


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

try:

  with open(save_path + 'data.pkl', 'r') as f:
    data = pickle.dump(f)
  with open(save_path + 'data_test.pkl', 'r') as f:
    data_test = pickle.load(f)
  with open(save_path + 'count.pkl', 'r') as f:
    count = pickle.load(f)
  with open(save_path + 'dictionary.pkl', 'r') as f:
    dictionary = pickle.load(f)
  with open(save_path + 'reverse_dictionary.pkl', 'r') as f:
    reverse_dictionary = pickle.load(f)
  with open(save_path + 'count_test.pkl', 'r') as f:
    count_test = pickle.load(f)
  with open(save_path + 'dictionary_test.pkl', 'r') as f:
    dictionary_test = pickle.load(f)
  with open(save_path + 'reverse_dictionary_test.pkl', 'r') as f:
    reverse_dictionary_test = pickle.load(f)

except:

  # Read the data into a list of strings.
  def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

  vocabulary = read_data(filename)
  print('Data size', len(vocabulary))

  vocabulary_size = 10000

  splits = len(vocabulary) // 1000
  train_set, test_set = [], []
  for i in range(500):
    train_set += vocabulary[(2*i)*splits:(2*i+1)*splits]
    test_set += vocabulary[(2*i+1)*splits:(2*i+2)*splits]

  del vocabulary

  def build_dataset(words, n_words, with_UNK = True, shuffle = False, count = None):
    """Process raw inputs into a dataset."""
    if count is None:
      if with_UNK:
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
      else:
        count = []
        count.extend(collections.Counter(words).most_common(n_words))

      if shuffle:
        count = np.random.permutation(count)
      else:
        count = count
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
        data.append(index)
      else:
        index = dictionary['UNK']
        unk_count += 1
        if with_UNK:
          data.append(index)
    if with_UNK:
      count[dictionary['UNK']][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary



  data, count, dictionary, reverse_dictionary = build_dataset(train_set,
    vocabulary_size)
  data_test, count_test, dictionary_test, reverse_dictionary_test = build_dataset(
    test_set, vocabulary_size, shuffle = False, count = count)
  vocabulary_size = min(vocabulary_size, len(count))

  # save data
  save_path = 'dat_{}'.format(vocabulary_size)
  sp.check_output('mkdir -p {}'.format(save_path), shell=True)
  save_path = './dat_{}/'.format(vocabulary_size)
  with open(save_path + 'data.pkl', 'w') as f:
    pickle.dump(data, f)
  with open(save_path + 'data_test.pkl', 'w') as f:
    pickle.dump(data_test, f)
  with open(save_path + 'count.pkl', 'w') as f:
    pickle.dump(count, f)
  with open(save_path + 'dictionary.pkl', 'w') as f:
    pickle.dump(dictionary, f)
  with open(save_path + 'reverse_dictionary.pkl', 'w') as f:
    pickle.dump(reverse_dictionary, f)
  with open(save_path + 'count_test.pkl', 'w') as f:
    pickle.dump(count_test, f)
  with open(save_path + 'dictionary_test.pkl', 'w') as f:
    pickle.dump(dictionary_test, f)
  with open(save_path + 'reverse_dictionary_test.pkl', 'w') as f:
    pickle.dump(reverse_dictionary_test, f)


count = count[:vocabulary_size]


def build_cooccurance_dict(data, count, dictionary, reverse_dictionary, skip_window):
  cooccurance_count = collections.defaultdict(collections.Counter)
  for idx, center_word in enumerate(data):
    center_word_id = center_word
    if idx >= skip_window - 1 and  idx < len(data) - skip_window:
      for i in range(skip_window):
        cooccurance_count[center_word_id][data[idx-i-1]] += 1
        cooccurance_count[center_word_id][data[idx+i+1]] += 1
    elif idx < skip_window - 1:
      for i in range(skip_window):
        cooccurance_count[center_word_id][data[idx+i+1]] += 1
      for i in range(idx):
        cooccurance_count[center_word_id][data[i]] += 1
    else:
      for i in range(skip_window):
        cooccurance_count[center_word_id][data[idx-i-1]] += 1
      for i in range(idx+1, len(data)):
        cooccurance_count[center_word_id][data[i]] += 1
  return cooccurance_count

skip_window = 5
try:
  with open(save_path + 'cooccur.pkl', 'r') as f:
    cooccur = pickle.load(f)
  with open(save_path + 'cooccur_test.pkl', 'r') as f:
    cooccur_test = pickle.load(f)
  with open(save_path + 'cooccur_matrix.pkl', 'r') as f:
    cooccur_matrix = pickle.load(f)
  cooccur_test_matrix = np.zeros([vocabulary_size, vocabulary_size])
  with open(save_path + 'cooccur_matrix_test.pkl', 'r') as f:
    cooccur_test_matrix = pickle.load(f)
  del data
  del data_test

except:
  cooccur = build_cooccurance_dict(data, count, dictionary, reverse_dictionary, skip_window)
  with open(save_path + 'cooccur.pkl', 'w') as f:
    pickle.dump(cooccur, f)


  #---------------------------build for second part of data---------------------
  cooccur_test = build_cooccurance_dict(data_test, count_test, 
    dictionary_test, reverse_dictionary_test, skip_window)

  with open(save_path + 'cooccur_test.pkl', 'w') as f:
    pickle.dump(cooccur_test, f)


  cooccur_matrix = np.zeros([vocabulary_size, vocabulary_size])
  for k1, v1 in cooccur.iteritems():
    for k2, v2 in v1.iteritems():
      cooccur_matrix[k1, k2] = v2
  del data
  with open(save_path + 'cooccur_matrix.pkl', 'w') as f:
    pickle.dump(cooccur_matrix, f)
  cooccur_test_matrix = np.zeros([vocabulary_size, vocabulary_size])
  for k1, v1 in cooccur_test.iteritems():
    for k2, v2 in v1.iteritems():
      cooccur_test_matrix[k1, k2] = v2
  del data_test

  with open(save_path + 'cooccur_matrix_test.pkl', 'w') as f:
    pickle.dump(cooccur_test_matrix, f)


Nij = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    Nij[i,j] += cooccur[i][j]
Ni = np.zeros(vocabulary_size)
for item in count:
  Ni[dictionary[item[0]]] = item[1]

tot = np.sum(Nij)
print(tot)
print(np.sum(Ni))
Pij = Nij / tot 
Pi = Ni / np.sum(Ni)
print(np.sum(Pij))
print(np.sum(Pi))


PMI = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    if Pi[i] * Pi[j] > 0 and Pij[i, j] > 0:
      PMI[i,j] = np.log(Pij[i,j] / (Pi[i] * Pi[j]))


Nij_test = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    Nij_test[i,j] += cooccur_test[i][j]
Ni_test = np.zeros(vocabulary_size)
for item in count_test:
  Ni_test[dictionary[item[0]]] = item[1]

tot = np.sum(Nij_test)
print(tot)
print(np.sum(Ni_test))
Pij_test = Nij_test / tot 
Pi_test = Ni_test / np.sum(Ni_test)
print(np.sum(Pij_test))
print(np.sum(Pi_test))


PMI_test = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    if Pi_test[i] * Pi_test[j] > 0 and Pij_test[i, j] > 0:
      PMI_test[i,j] = np.log(Pij_test[i,j] / (Pi_test[i] * Pi_test[j]))


diff = PMI - PMI_test
print("mean is {}".format(np.mean(diff)))
print("est std is {}".format(0.5 * np.std(diff)))

