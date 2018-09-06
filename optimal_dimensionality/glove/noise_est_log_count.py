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

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='estimation of noise given a corpus')
  parser.add_argument('--filename', required=True, type=str, help='vocabulary siz3')
  parser.add_argument('--vocabulary_size', default=10000, type=int, help='vocabulary siz3')
  parser.add_argument('--window_size', default=5, type=int, help='window size')
  parser.add_argument('intrinsic_test', action='store_false')
  args = parser.parse_args()
  filename = args.filename
  vocabulary_size = args.vocabulary_size
  skip_window = args.window_size
  save_path = 'dat_{}/'.format(vocabulary_size)

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
    vocabulary = read_data(filename)
    vocabulary_size = args.vocabulary_size

    splits = len(vocabulary) // 1000
    train_set, test_set = [], []
    for i in range(500):
      train_set += vocabulary[(2*i)*splits:(2*i+1)*splits]
      test_set += vocabulary[(2*i+1)*splits:(2*i+2)*splits]

    del vocabulary


    data, count, dictionary, reverse_dictionary = build_dataset(train_set,
      vocabulary_size)
    data_test, count_test, dictionary_test, reverse_dictionary_test = build_dataset(
      test_set, vocabulary_size, shuffle = False, count = count)
    vocabulary_size = min(vocabulary_size, len(count))

    # save data
    sp.check_output('mkdir -p {}'.format(save_path), shell=True)
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


  skip_window = args.window_size
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

  log_count = np.log(1 + Nij)

  Nij_test = np.zeros([vocabulary_size, vocabulary_size])
  for i in range(vocabulary_size):
    for j in range(vocabulary_size):
      Nij_test[i,j] += cooccur_test[i][j]
  log_count_test = np.log(1 + Nij_test)

  diff = log_count - log_count_test
  print("mean is {}".format(np.mean(diff)))
  print("est std is {}".format(0.5 * np.std(diff)))
  with open("param.yml", "w") as f:
      f.write("sigma: {}\n".format(0.5 * np.std(diff)))
      f.write("alpha: {}\n".format(0.5)) #symmetric factorization
      f.write("data: {}".format("text8_logcount"))
