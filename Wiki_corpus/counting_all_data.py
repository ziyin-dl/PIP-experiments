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

split = int(1 * len(vocabulary))
train_set = vocabulary[:split]
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



data, count, dictionary, reverse_dictionary = build_dataset(train_set,
  vocabulary_size)#, existing = test_set_vocab)
vocabulary_size = max(vocabulary_size, len(count))

# save data
save_path = './dat_enriched_{}/'.format(vocabulary_size)
if not os.path.exists(save_path):
  os.makedirs(save_path)

with open(save_path + 'count.pkl', 'w') as f:
  pickle.dump(count, f)
with open(save_path + 'dictionary.pkl', 'w') as f:
  pickle.dump(dictionary, f)
with open(save_path + 'reverse_dictionary.pkl', 'w') as f:
  pickle.dump(reverse_dictionary, f)

count = count[:vocabulary_size]

print('Top 10 most freq word for train data: {}'.format(count[:10]))
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

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
cooccur = collections.defaultdict(collections.Counter)

print('begin building the initial cooccurance data.')
build_cooccurance_dict(data, dictionary, reverse_dictionary, skip_window, cooccur)

print('finished phase 1. build the rest of the dataset.')
data = []
cnt = 0
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
      build_cooccurance_dict(data, dictionary, reverse_dictionary, skip_window, cooccur)
      cnt = 0
      data = []

with open(save_path + 'cooccur.pkl', 'w') as f:
  pickle.dump(cooccur, f)



cooccur_matrix = np.zeros([vocabulary_size, vocabulary_size])
for k1, v1 in cooccur.iteritems():
  for k2, v2 in v1.iteritems():
    cooccur_matrix[k1, k2] = v2
del data
with open(save_path + 'cooccur_matrix.pkl', 'w') as f:
  pickle.dump(cooccur_matrix, f)


exit()
