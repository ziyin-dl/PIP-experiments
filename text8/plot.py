from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import subprocess as sp
import math
import os
import random
import cPickle as pickle
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import bisect
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

parser = argparse.ArgumentParser(description='experiment with the effect of signal/noise for the invariant spaces.')
parser.add_argument('--folder', required = True, type=str)

args = parser.parse_args()
dir_name = args.folder

telescoping_expansions = {}

sim_tests_dict = {}

for root, dirs, files in os.walk(dir_name):
  for file in files:
    if '.pkl' not in file:
      continue
    print('{}/{}'.format(root, file))
    # naming convensions, remove .pkl then...
    alpha = float(file[:-4].split('_')[-1])
    with open(os.path.join(root, file), "r") as f:
      sim_tests = pickle.load(f)
      sim_tests_dict[alpha] = sim_tests
      print(alpha)
      keys = sim_tests.keys()

for test_key in keys:
  test_set_name = test_key.split('.')[0]
  alpha_keys = sorted(sim_tests_dict.keys())
  for k in alpha_keys:
    argmax = 0
    max_val = 0
    v = sim_tests_dict[k][test_key]
    for idx, val in enumerate(v):
      if val > max_val:
        max_val = val
        argmax = idx
    print('test {} with alpha={} has argmax {} with value {}'.format(test_set_name, k, argmax, max_val))

for test_key in keys:
  test_set_name = test_key.split('.')[0]
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  alpha_keys = sorted(sim_tests_dict.keys())
  print(alpha_keys)
  for k in alpha_keys:
    v = sim_tests_dict[k][test_key]
    plt.plot(v, label=r'$\alpha$={}'.format(k))
  lgd = ax.legend(loc='upper right')
  plt.title(r'Correlation with Human Labels for Different $\alpha$, Test Set is {}'.format(test_set_name))
  plt.xlabel('Dimensions')
  plt.ylabel('Correlation with Human Labels')
  fig.savefig('{}/corr_{}.pdf'.format(dir_name, test_set_name), bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close()

