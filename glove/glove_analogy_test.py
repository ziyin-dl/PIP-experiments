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

def read_analogies(analogy_filename, dictionary):
  questions = []
  questions_skipped = 0
  with open(analogy_filename, "rb") as analogy_f:
    for line in analogy_f:
      if line.startswith(b":"):  # Skip comments.
        continue
      words = line.strip().lower().split(b" ")
      ids = [dictionary.get(w.strip()) for w in words]
      if None in ids or len(ids) != 4:
        questions_skipped += 1
      else:
        questions.append(np.array(ids))
  print("Eval analogy file: ", analogy_filename)
  print("Questions: ", len(questions))
  print("Skipped: ", questions_skipped)
  return np.array(questions, dtype=np.int32)

def predict(analogy_first3, embed, dictionary):
  analogy_a = analogy_first3[0]
  analogy_b = analogy_first3[1]
  analogy_c = analogy_first3[2]
  a_emb = embed[analogy_a, :]
  b_emb = embed[analogy_b, :]
  c_emb = embed[analogy_c, :]
  target  = c_emb + (b_emb - a_emb)
  dist = target.dot(embed.T)
  predicted = np.argsort(dist, axis=1)[:, -4:]
  print(predicted.shape)
  return predicted

def eval(analogy_questions, embed, dictionary):
  correct = 0
  try:
    total = analogy_questions.shape[0]
  except AttributeError as e:
    raise AttributeError("Need to read analogy questions.")
  reverse_dictionary = {}
  for k, v in dictionary.iteritems():
    reverse_dictionary[v] = k
  start = 0
  while start < total:
    limit = start + 2500
    sub = analogy_questions[start:limit, :]
    idx = predict([sub[:, 0], sub[:, 1], sub[:, 2]], embed, dictionary)
    start = limit
    for question in xrange(sub.shape[0]):
      print("{} - {} = {} - ({})".format(reverse_dictionary[sub[question, 0]],
        reverse_dictionary[sub[question, 1]], reverse_dictionary[sub[question, 2]], reverse_dictionary[sub[question, 3]]))
      print("top fours are {}, {}, {}, {}".
          format(reverse_dictionary[idx[question, 3]], reverse_dictionary[idx[question, 2]],
            reverse_dictionary[idx[question, 1]], reverse_dictionary[idx[question, 0]]))
      for j in xrange(4):
        if idx[question, j] == sub[question, 3]:
          # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
          correct += 1
          break
        elif idx[question, j] in sub[question, :3]:
          # We need to skip words already in the question.
          continue
        else:
          # The correct label is not the precision@1
          break
  print()
  print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                            correct * 100.0 / total))
  return correct*1.0 / total


def load_data(dir_name):
  """dictionary is to map words to indices, data is to map dimensions to embedding matrices"""
  dictionary = collections.OrderedDict()
  data = collections.OrderedDict()
  test_subset = None
  test_subset = {300, 400, 500}
  cnt = 0
  for root, dirs, files in os.walk(dir_name):
    for file in files:
      if '.txt' not in file or 'vectors' not in file:
        continue
      dim = int(file[7:].split('.')[0])
      if test_subset and dim not in test_subset:
        continue
      cnt += 1
      print('{}/{}'.format(root, file))
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
  parser.add_argument("--testfile", required=True, type=str)
  args = parser.parse_args()
  dictionary, data = load_data(args.folder)
  test = read_analogies(args.testfile, dictionary)
  dir_name = 'analogy_test_results'
  sp.check_output('mkdir -p {}'.format(dir_name), shell=True)

  dims = sorted(data.keys())
  analogy_scores = []
  test_file = args.testfile
  test_rows = test
  best_dim = [0, 0]
  for dim in dims:
    print("evaluating dimension {}".format(dim))
    embeddings = data[dim]
    score = eval(test, embeddings, dictionary)
    analogy_scores.append(score)
    if score > best_dim[1]:
      best_dim = [dim, score]

  with open('{}/analogy_test_result.pkl'.format(dir_name), 'w') as f:
    pickle.dump(analogy_scores, f)

  plt.plot(dims, analogy_scores, label=test_file.split('.')[0])
  plt.legend(loc='lower right')
  plt.title('Analogy Performance vs Embedding Size')
  plt.ylabel('top-k hit rate')
  plt.xlabel('dimensions')
  print("best for {} is at {} with {}".format(test_file.split('.')[0], best_dim[0], best_dim[1]))
  plt.savefig('{}/analogy_scores.pdf'.format(dir_name))
  plt.close()
