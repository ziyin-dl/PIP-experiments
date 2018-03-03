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
import string
import re

filename = 'sick2014/SICK_train.txt'

def charFilter(my_string):
  return re.sub(r"[^A-Za-z]+", '', my_string) 

# Read the data into a list of strings.
def read_data(filename):
  # format: pair_ID\t sent_A\t sentB\t score\t judgement
  data1 = []
  data2 = []
  score = []
  vocabulary = []
  with open(filename, 'r') as f:
    for idx, line in enumerate(f):
      row = line.split('\t')
      if idx == 0:
        continue
      data1.append(filter(None, map(charFilter, row[1].strip().lower().translate(None, string.punctuation).split(' '))))
      data2.append(filter(None, map(charFilter, row[2].strip().lower().translate(None, string.punctuation).split(' '))))
      # print(data1[-1])
      # print(data2[-1])
      vocabulary += data1[-1]
      vocabulary += data2[-1]
      score.append(float(row[3]))
  return vocabulary, data1, data2, score


vocabulary, set1, set2, scores = read_data(filename)

vocabulary_size = 10000




def build_dictionary(words, n_words, with_UNK = True, shuffle = False, dictionary = None, existing = None):
  """Process raw inputs into a dataset."""

  if dictionary is None:
    if with_UNK:
      top_n_common = collections.Counter(words).most_common(n_words - 1)
      if len(top_n_common) == n_words - 1:
        count = [['UNK', -1]]
      else:
        count = []
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
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return dictionary, reversed_dictionary


def build_dataset(sets, dictionary, with_UNK = True):
  data = []
  for sent in sets:
    tokenized_sent = []
    for word in sent:
      if word in dictionary:
        index = dictionary[word]
        tokenized_sent.append(index)
      else:
        if with_UNK:
          index = dictionary['UNK']
          tokenized_sent.append(index)
    data.append(tokenized_sent)
  return data

dictionary, reverse_dictionary = build_dictionary(vocabulary, vocabulary_size)
vocabulary_size = min(vocabulary_size, len(dictionary))

tokenized_set1 = build_dataset(set1, dictionary)
tokenized_set2 = build_dataset(set2, dictionary)


# save data
save_path = './dat_{}/'.format(vocabulary_size)
if not os.path.exists(save_path):
  os.makedirs(save_path)

with open(save_path + 'dictionary.pkl', 'w') as f:
  pickle.dump(dictionary, f)
with open(save_path + 'reverse_dictionary.pkl', 'w') as f:
  pickle.dump(reverse_dictionary, f)


print('Sample data', set1[0], [reverse_dictionary[i] for i in tokenized_set1[0]])

def build_cooccurance_dict(set1, dictionary):
  tf_matrix = np.zeros((len(dictionary), len(set1)))
  for i in range(len(set1)):
    for item in set1[i]:
      tf_matrix[item, i] += 1.0
  return tf_matrix / np.reshape(np.sum(tf_matrix, axis = 0), (1, tf_matrix.shape[1]))

def tf_idf_matrix(tf_matrix):
  N = tf_matrix.shape[1]
  df = np.count_nonzero(tf_matrix, axis = 1)
  print('{} many zero entries in df'.format(len(df) - np.count_nonzero(df)))
  df[df == 0] = 1
  idf = np.log(1 + N / df)
  # so idf will be a vector of size tf_matrix.shape[0]
  
  tf_idf_matrix = np.expand_dims(idf, axis = 1) * tf_matrix
  print("idf shape={}".format(np.expand_dims(idf, axis = 1).shape))
  print("tf matrix shape = {}".format(tf_matrix.shape))
  return tf_idf_matrix

 
# they are paired dataset; hence should have equal length.
# now, concatinate them into a larger one; with 1st half and 2nd half 
tf_matrix1 = build_cooccurance_dict(tokenized_set1, dictionary)
tf_matrix2 = build_cooccurance_dict(tokenized_set2, dictionary)
tf_idf1 = tf_idf_matrix(tf_matrix1)
tf_idf2 = tf_idf_matrix(tf_matrix2)
print(np.linalg.norm(tf_idf1 - tf_idf2, 'fro') / np.sqrt(len(set1) * len(dictionary)))
tf_matrix = build_cooccurance_dict(tokenized_set1 + tokenized_set2, dictionary)
scores_columns = {}
for idx, score in enumerate(scores):
  scores_columns[(idx, idx + len(scores))] = score

tf_idf = tf_idf_matrix(tf_matrix)
U, D, V = np.linalg.svd(tf_idf)
with open('tfidf_sv.pkl', 'w') as f:
  pickle.dump(D, f)

# tf_idf = tf_matrix

exit()
with open(save_path + 'tf_idf.pkl', 'w') as f:
  pickle.dump(tf_idf, f)

with open(save_path + 'scores.pkl', 'w') as f:
  pickle.dump(scores_columns, f)


