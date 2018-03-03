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
parser.add_argument('--sigma', required = True, default = 0.1, type = float, help='noise level')
parser.add_argument('--alpha', required = True, default = 0, type = float)
parser.add_argument('--num_runs', default = 10, type = int)
parser.add_argument('--data', default = 'wikipedia', type = str)
parser.add_argument('--conf_int', nargs = '+', default = 0.2, type = float)

args = parser.parse_args()
alpha = args.alpha
sigma = args.sigma
conf_ints = args.conf_int
num_runs = args.num_runs

def soft_threshold(x, tau):
    if x > tau:
        return x - tau
    else:
        return 0

# load the empirical singular values from wikipedia truncated top 10000 words
with open('sv_{}.pkl'.format(args.data), 'r') as f:
    D = pickle.load(f)

def generate_random_orthogonal_matrix(shape):
    assert len(shape) == 2
    assert shape[0] >= shape[1]
    X = np.random.normal(0, 1, shape)
    U, _, _ = np.linalg.svd(X, full_matrices = False)
    return U
    
n = len(D)
# n = 1000
shape = (n, n)
D = map(lambda x: soft_threshold(x, 2 * sigma * np.sqrt(len(D))), D)
for i in range(len(D)):
    if D[i] == 0:
        rank = i
        break
print('rank is {}'.format(rank))

dir_name = '{}/alpha_{}_sigma_{}_rank_{}'.format(args.data, alpha, sigma, rank)
local_dir_name = '/hdd/zyin/dropbox/matrix_exp/{}'.format(dir_name)
sp.check_output('mkdir -p {}'.format(dir_name), shell=True)
try:
  sp.check_output('mkdir -p {}'.format(local_dir_name), shell=True)
except:
  pass
plt.plot(D)
plt.savefig('{}/sv_X.pdf'.format(dir_name))
plt.close()

try:
  with open('{}/results.pkl'.format(dir_name), 'r') as f:
    [frobenius_list_est_to_gt_dict, var_sv_ub_dict,
      var_subspace_ub_dict,
      var_sv_gt_dict,
      var_subspace_gt_dict,
      telescoping_ub_dict,
      telescoping_expansion_dict,
      gt_ub_dict] = pickle.load(f)
    results = pickle.load(f)
    for k, v in results:
      print('{}: {}'.format(k,v))
except Exception as e:
  print(e)


'''first plot: overall bound quality vs ground truth'''
fig = plt.figure()
ax  = fig.add_subplot(111)
level_sets = []
level_sets_endpoints = []
colors = ['r', 'g', 'b', 'k']
for conf_int in conf_ints:
  for i in range(len(telescoping_expansion_dict)):
    frobenius_list_est_to_gt = frobenius_list_est_to_gt_dict[i]
    gt_min_x = np.argmin(frobenius_list_est_to_gt)
    gt_min_y = frobenius_list_est_to_gt[gt_min_x]
    # calculate in what interval does error stay within x% of optimal
    conf_int_ub = gt_min_y + conf_int * (frobenius_list_est_to_gt[0] - gt_min_y)
    level_sets.append(conf_int_ub)
    conf_endpoints = [0, len(frobenius_list_est_to_gt) - 1]
    for idx in range(len(frobenius_list_est_to_gt)):
      if frobenius_list_est_to_gt[idx] < conf_int_ub:
        conf_endpoints[0] = idx
        break
    for idx in reversed(range(len(frobenius_list_est_to_gt))):
      if frobenius_list_est_to_gt[idx] < conf_int_ub:
        conf_endpoints[1] = idx
        break
    print('optimal is at {}'.format(gt_min_x))
    print('between {} and {}, the PIP error is not more than {}% of optimal'.format(
      conf_endpoints[0], conf_endpoints[1], 100 * conf_int))
    level_sets_endpoints.append(conf_endpoints)
for i in range(len(telescoping_expansion_dict)):
  frobenius_list_est_to_gt = frobenius_list_est_to_gt_dict[i]
  if i == 0:
    ax.plot(frobenius_list_est_to_gt, 'aqua', label = r'grount truth PIP error')
  else:
    ax.plot(frobenius_list_est_to_gt, 'aqua')

assert(len(colors) == len(level_sets))
for level, pct, endpoints, color in zip(level_sets, conf_ints, level_sets_endpoints, colors):
  xlim = ax.get_xlim()
  plt.axhline(y=level, xmin=(endpoints[0] - xlim[0]) / (xlim[1] - xlim[0]), 
    xmax=(endpoints[1] - xlim[0]) / (xlim[1]-xlim[0]),
    label='+{}'.format(pct * 100) + r'$\%$', ls='--', c=color)
lgd = ax.legend(loc='upper right')
plt.xlabel('Dimensions')
plt.ylabel('PIP Error')
plt.title(r'PIP Error Ground Truth'.format(alpha))
fig.savefig('{}/pip_gt_{}.pdf'.format(dir_name, alpha), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()