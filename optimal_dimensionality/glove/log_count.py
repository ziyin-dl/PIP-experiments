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
from six.moves import xrange  # pylint: disable=redefined-builtin
import bisect
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import subprocess as sp

# Read the data into a list of strings.
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
  parser = argparse.ArgumentParser(description='construct pmi matrix from corpus')
  parser.add_argument('--filename', required=True, type=str, help='vocabulary siz3')
  parser.add_argument('--vocabulary_size', default=10000, type=int, help='vocabulary siz3')
  parser.add_argument('--window_size', default=5, type=int, help='window size')
  parser.add_argument('intrinsic_test', action='store_false')
  args = parser.parse_args()
  
  filename = args.filename
  vocabulary_size = args.vocabulary_size
  skip_window = args.window_size
  vocabulary = read_data(filename)
  train_set = vocabulary

  data, count, dictionary, reverse_dictionary = build_dataset(train_set,
    vocabulary_size)
  vocabulary_size = min(vocabulary_size, len(count))

  print('Top 10 most freq word for train data: {}'.format(count[:10]))
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  cooccur = build_cooccurance_dict(data, count, dictionary, reverse_dictionary, skip_window)
  # Do SVD
  Nij = np.zeros([vocabulary_size, vocabulary_size])
  for i in range(vocabulary_size):
    for j in range(vocabulary_size):
      Nij[i,j] += cooccur[i][j]
  log_count = np.log(1 + Nij)

  U, D, V = np.linalg.svd(log_count)
  plt.plot(D)
  plt.savefig('sv_PPMI_top{}_logcount.pdf'.format(vocabulary_size))
  plt.close()

  with open('sv_text8_logcount.pkl', 'w') as f:
    pickle.dump(D, f)

  if args.intrinsic_test:
    test_set_files = ['wordsim353.csv', 'mturk771.csv']
    test_rows_list = [load_test_file(test_set_file) for test_set_file in test_set_files]

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
