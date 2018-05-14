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
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import subprocess as sp

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'
parser = argparse.ArgumentParser(description='experiment with the effect of signal/noise for the invariant spaces.')
parser.add_argument('--alpha', nargs='+', required = True)
args = parser.parse_args()

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


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))


vocabulary_size = 10000
train_set = vocabulary


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
vocabulary_size = min(vocabulary_size, len(count))

print('Top 10 most freq word for train data: {}'.format(count[:10]))
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

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
cooccur = build_cooccurance_dict(data, count, dictionary, reverse_dictionary, skip_window)
# Do SVD
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

PMI = np.zeros([vocabulary_size, vocabulary_size])
for i in range(vocabulary_size):
  for j in range(vocabulary_size):
    if Pi[i] * Pi[j] > 0 and Pij[i, j] > 0:
      PMI[i,j] = np.log(Pij[i,j] / (Pi[i] * Pi[j]))


U, D, V = np.linalg.svd(PMI)
plt.plot(D)
plt.savefig('sv_PMI_top{}.pdf'.format(vocabulary_size))
plt.close()

with open('sv_text8_pmi.pkl', 'w') as f:
  pickle.dump(D, f)

