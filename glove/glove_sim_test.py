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


def load_data(dir_name):
  """dictionary is to map words to indices, data is to map dimensions to embedding matrices"""
  dictionary = collections.OrderedDict()
  data = collections.OrderedDict()
  for root, dirs, files in os.walk(dir_name):
    for file in files:
      if '.txt' not in file:
        continue
      print('{}/{}'.format(root, file))
      dim = int(file[7:].split('.')[0])
      print("loading dim {}".format(dim))
      with open('{}/{}'.format(root, file), 'r') as f:
        ordered_index = []
        ordered_data = []
        for line in f:
          row = line.strip().split(' ');
          wd = row[0]
          vec = [float(x) for x in row[1:]]
          if wd not in dictionary:
            dictionary[wd] = len(dictionary)
          ordered_index.append(dictionary[wd])
          ordered_data.append(vec)
        ordered_data = [ordered_data[i] for i in ordered_index]
        data[dim] = np.array(ordered_data)
  return dictionary, data

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", required=True, type=str)
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

  dictionary, data = load_data(args.folder)
  dims = sorted(data.keys())

  for dim in dims:
    print("evaluating dimension {}".format(dim))
    embeddings = data[dim]
    for test_file, test_rows in zip(test_set_files, test_rows_list):
      score1, score2 = test(test_rows, dictionary, embeddings)
      corr = np.corrcoef(score1, score2)[0,1]
      sim_tests[test_file].append(corr)
      if corr > sim_best_dim[test_file][1]:
        sim_best_dim[test_file] = (dim, corr)
  with open('{}/sim_test_result.pkl'.format(dir_name), 'w') as f:
    pickle.dump(sim_tests, f)
  for k, v in sim_tests.iteritems():
    plt.plot(dims, v, label=k.split('.')[0])
    plt.legend(loc='lower right')
    plt.title('Similarity Task Performance vs Embedding Size')
    plt.ylabel('correlation with hunam labels')
    plt.xlabel('dimensions')
    print("best for {} is at {} with {}".format(k.split('.')[0], sim_best_dim[k][0], sim_best_dim[k][1]))
    plt.savefig('{}/sim_scores_{}.pdf'.format(dir_name, k.split('.')[0]))
    plt.close()
