from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import cPickle as pickle
import numpy as np
import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import subprocess as sp

def load_test_file(test_set_file):
  rows = []
  with open(test_set_file, 'r') as f:
    for idx, line in enumerate(f):
      row = line.strip().replace(';', ',').split(',')
      rows.append(row)
    return rows

def test(rows, dictionary, embeddings_U):
  score1, score2 = [], []
  for idx, row in enumerate(rows):
    if row[0] in dictionary and row[1] in dictionary:
      score1.append(float(row[2]))
      word1 = dictionary[row[0]]
      word2 = dictionary[row[1]]
      score2.append(embeddings_U[word1,:].dot(embeddings_U[word2,:].T) / (np.linalg.norm(embeddings_U[word1,:], 2) * np.linalg.norm(embeddings_U[word2,:], 2)))
  return score1, score2


def load_data(dir_name, dictionary = None):
  """dictionary is to map words to indices, data is to map dimensions to embedding matrices"""
  if dictionary is None:
    dictionary = collections.OrderedDict()
  data = collections.OrderedDict()
  file_cnt = 0
  for root, dirs, files in os.walk(dir_name):
    for file in files:
      # if not (('300.txt' in file) or ('1000.txt' in file)):
      if not ('.txt' in file and 'vector' in file):
        continue
      print('{}/{}'.format(root, file))
      dim = int(file[7:].split('.')[0])
      print("loading dim {}".format(dim))
      with open('{}/{}'.format(root, file), 'r') as f:
        ordered_index = {}
        ordered_data = []
        for line in f:
          row = line.strip().split(' ');
          wd = row[0]
          if wd == "UNK":
            wd = "<unk>"
          vec = [float(x) for x in row[1:]]
          if wd not in dictionary:
            dictionary[wd] = len(dictionary)
          ordered_index[dictionary[wd]] = vec
          ordered_data.append(vec)
        print(len(ordered_data))
        print(len(ordered_index))
        ordered_data = [ordered_index[i] for i in range(len(ordered_index))]
        data[dim] = np.array(ordered_data)
        file_cnt += 1
        # if file_cnt >= 10:
        #   break
  return dictionary, data

def PIP(E1, normalize=True):
  if normalize:
    E1_len = np.sqrt(np.sum(E1 * E1, axis=1))
    print(np.mean(E1_len))
    E1_norm = E1 / np.mean(E1_len)
    print(np.mean(np.sqrt(np.sum(E1_norm * E1_norm, axis=1))))
  else:
    E1_norm = E1
  PIP1 = E1_norm.dot(E1_norm.T)
  return PIP1

def PIP_loss(PIP1, PIP2):
  assert(PIP1.shape[0] == PIP2.shape[0])
  wd_idx = 9991
  sim1 = list(reversed(np.argsort(PIP1[wd_idx,:])[-10:]))
  # print("closest to {}".format(reverse_dictionary[wd_idx]))
  # print("{}".format(" ".join(map(str, zip([reverse_dictionary[x] for x in sim1], [PIP1[wd_idx, x] for x in sim1])))))
  sim2 = list(reversed(np.argsort(PIP2[wd_idx,:])[-10:]))
  # print("closest to {}".format(reverse_dictionary[wd_idx]))
  # print("{}".format(" ".join(map(str, zip([reverse_dictionary[x] for x in sim2], [PIP1[wd_idx, x] for x in sim2])))))

  PIP_loss = np.linalg.norm(PIP1 - PIP2, 'fro')
  PIP1_norm = np.linalg.norm(PIP1, 'fro')
  PIP2_norm = np.linalg.norm(PIP2, 'fro')
  print("PIP for E1 has power {}".format(PIP1_norm ** 2))
  print("PIP for E2 has power {}".format(PIP2_norm ** 2))
  nsr = (PIP_loss ** 2) / (PIP1_norm * PIP2_norm)
  print("PIP loss has power {}, nsr is {}".format(PIP_loss ** 2, nsr))
  return PIP_loss, nsr

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", nargs='+', required=True, type=str)
  args = parser.parse_args()
  test_set_files = ['wordsim353.csv', 'mturk771.csv', 'rg65.csv', 'mc91.csv']
  test_rows_list = [load_test_file(test_set_file) for test_set_file in test_set_files]
  sim_tests = {}
  for test_file in test_set_files:
    sim_tests[test_file] = []
  sim_best_dim = {}
  for test_file in test_set_files:
    sim_best_dim[test_file] = (0, 0)

  dir_name = 'test_results'
  sp.check_output('mkdir -p {}'.format(dir_name), shell=True)

  dictionary = None
  procedure_data = collections.OrderedDict()
  for folder in args.folder:
    dictionary, data = load_data(folder, dictionary)
    procedure_data[folder] = data
  dims = sorted(data.keys())
  reverse_dictionary = {}
  for k, v in dictionary.iteritems():
    reverse_dictionary[v] = k
  global reverse_dictionary

  embeds = []
  for dim in dims:
    embeds_dim = []
    for k, v in procedure_data.iteritems():
      embeds_dim.append(v[dim])
      print("{}/{}: {}".format(k, dim, v[dim].shape))
    embeds.append(embeds_dim)

  keys = procedure_data.keys()
  losses, nsrs = [], []
  for idx, dim in enumerate(dims):
    PIP1 = PIP(embeds[idx][0], False)
    PIP2 = PIP(embeds[idx][1], False)
    loss, nsr = PIP_loss(PIP1, PIP2)
    losses.append(loss)
    nsrs.append(nsr)
    print("dim {}: loss is {}, noise-signal ratio is {}".
        format(dim, loss, nsr))
    with open("crossval.pkl", 'w') as f:
      pickle.dump([dims, losses, nsrs], f)
