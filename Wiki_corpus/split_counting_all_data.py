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


def read_test_sets(directory):
  words = {}
  for filename in os.listdir(directory):
    if filename.endswith(".csv"):
      with open(os.path.join(directory, filename), 'r') as f:
        for idx, line in enumerate(f):
          row = line.split(',')
          if row[0] not in words:
            words[row[0]] = 1
          if row[1] not in words:
            words[row[1]] = 1
  print('{} many words'.format(len(words)))
  return words

chunks = 2
test_set_vocab = read_test_sets('./tests/')
print(test_set_vocab)


filename = 'wiki.en.text'
trial_read = 1800000


# Read the data into a list of strings.
def read_data(filename):
  data = []
  with open(filename, 'r') as f:
    for idx, line in enumerate(f):
      row = line.split()
      data += row
      if idx > trial_read:
        break
  return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

vocabulary_size = 10000

datasets = []

for i in range(1, chunks):
  last_split = int(1.0 / chunks * (i-1) * len(vocabulary))
  split = int(1.0 / chunks * i * len(vocabulary))
  datasets.append(vocabulary[last_split:split])
datasets.append(vocabulary[split:])
del vocabulary

def build_dataset(words, n_words, with_UNK = True, shuffle = False, dictionary = None, existing = None):
  """Process raw inputs into a dataset."""

  _is_return_count = False
  if dictionary is None:
    _is_return_count = True
    if with_UNK:
      top_n_common = collections.Counter(words).most_common(n_words - 1)
      count = [['UNK', -1]]
      count.extend(top_n_common)
    else:
      count = []
      top_n_common = collections.Counter(words).most_common(n_words)
      count.extend(top_n_common)

    if shuffle:
      count = np.random.permutation(count)
    count_words = map(lambda x: x[0], count) 
  
    if existing:
      existing_counter = collections.Counter()
      for word in words:
        if word in existing:
          existing_counter[word] += 1
      for word in existing_counter:
        if word not in count_words and existing_counter[word] > 1:
          count.append([word, 0])

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
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  if not _is_return_count:
    count = None
  return data, count, dictionary, reversed_dictionary

datas = []
dictionarys = []
reverse_dictionarys = []
for i in range(chunks):
  data, _, dictionary, reverse_dictionary = build_dataset(datasets[i], vocabulary_size)
  dictionarys.append(dictionary)
  reverse_dictionarys.append(reverse_dictionary)
  datas.append(data)
del datasets

# save data
save_path = './dat_splitted_{}/'.format(vocabulary_size)
if not os.path.exists(save_path):
  os.makedirs(save_path)

# with open(save_path + 'count.pkl', 'w') as f:
#   pickle.dump(count, f)
for i in range(chunks):
  with open(save_path + 'dictionary_{}.pkl'.format(i), 'w') as f:
    pickle.dump(dictionarys[i], f)
  with open(save_path + 'reverse_dictionary_{}.pkl'.format(i), 'w') as f:
    pickle.dump(reverse_dictionarys[i], f)


def build_cooccurance_dict(data, dictionary, reverse_dictionary, skip_window, cooccurance_count):
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
  testing_words = ['UNK', 'the', 'america', 'one', 'september']
  testing_indices = map(lambda x: dictionary[x], testing_words)
  for i in range(1, len(testing_indices)):
    print('number of cooccurrences between {} and {} is {}'.format(testing_words[i-1], testing_words[i], cooccurance_count[testing_indices[i-1]][testing_indices[i]]))

print('finished building the initial dataset.')
skip_window = 5
cooccurs = []
for i in range(chunks):
  cooccurs.append(collections.defaultdict(collections.Counter))

print('begin building the initial cooccurance data.')

for i in range(chunks):
  build_cooccurance_dict(datas[i], dictionarys[i], reverse_dictionarys[i], skip_window, cooccurs[i])

print('finished phase 1. build the rest of the dataset.')
data = []
cnt = 0
toggle = 0
with open(filename, 'r') as f:
  for idx, line in enumerate(f):
    if idx <= trial_read:
      continue
    row = line.split()
    data += row
    cnt += 1
    if cnt == 100000:
      print('this is epoch {}. Building dataset for this epoch.'.format((idx - trial_read) / 100000))
      data, _, _, _ = build_dataset(data, None, dictionary = dictionary)
      print('building coocurrances.')
      current_index = toggle % chunks
      build_cooccurance_dict(data, dictionarys[current_index], reverse_dictionarys[current_index], skip_window, cooccurs[current_index])
      cnt = 0
      toggle += 1
      data = []

for i in range(chunks):
  with open(save_path + 'cooccur_{}.pkl'.format(i), 'w') as f:
    pickle.dump(cooccurs[i], f)


cooccur_matrices = []

for i in range(chunks):
  cooccur_matrix = np.zeros([vocabulary_size, vocabulary_size])
  for k1, v1 in cooccurs[i].iteritems():
    for k2, v2 in v1.iteritems():
      cooccur_matrix[k1, k2] = v2
  cooccur_matrices.append(cooccur_matrix)

del data

for i in range(chunks):
  with open(save_path + 'cooccur_matrix_{}.pkl'.format(i), 'w') as f:
    pickle.dump(cooccur_matrices[i], f)


exit()
