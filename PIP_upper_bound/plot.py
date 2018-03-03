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


for root, dirs, files in os.walk(dir_name):
  for file in files:
    if '.pkl' not in file:
      continue
    print('{}/{}'.format(root, file))
    alpha = root.split('/')[1].split('_')[1]
    with open(os.path.join(root, file), "r") as f:
      [frobenius_list_est_to_gt_dict, var_sv_ub_dict,
        var_subspace_ub_dict,
        var_sv_gt_dict,
        var_subspace_gt_dict,
        telescoping_ub_dict,
        telescoping_expansion_dict,
        gt_ub_dict] = pickle.load(f)
      telescoping_expansions[alpha] = telescoping_expansion_dict[0]
try:
  print('done')
except:
  print('Error: no result file')
  exit()



'''first plot: overall bound quality vs ground truth'''
fig = plt.figure()
ax  = fig.add_subplot(111)

keys = sorted(telescoping_expansions.keys())
for k in keys:
  v = telescoping_expansions[k]
  v = map(lambda x: x / v[0], v)
  plt.plot(v, label=r'$\alpha$={}'.format(k))
lgd = ax.legend(loc='upper left')
plt.title(r'Relative PIP Error Predictions for different $\alpha$')
plt.xlabel('Dimensions')
plt.ylabel('Relative PIP Error')
fig.savefig('{}/bounds.pdf'.format(dir_name, alpha), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

