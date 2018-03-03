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
parser.add_argument('--alpha', required = True, default = 0, type = float, help='U.dot(Sigma^x), if we want to streth the axis accordingly')
parser.add_argument('--num_runs', default = 1, type = int)
parser.add_argument('--data', default = 'wikipedia', type = str)
parser.add_argument('--conf_int', default = 0.2, type = float)

args = parser.parse_args()
alpha = args.alpha
sigma = args.sigma
conf_int = args.conf_int
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
except:
  frobenius_list_est_to_gt_dict = []
  var_sv_ub_dict = []
  var_subspace_ub_dict = []
  var_sv_gt_dict = []
  var_subspace_gt_dict = []
  telescoping_ub_dict = []
  telescoping_expansion_dict = []
  gt_ub_dict = []

  for run_idx in range(num_runs):
    print('run {}'.format(run_idx))
    try:
      with open('{}/SVD_1.pkl'.format(local_dir_name), 'r') as f:
        U, D, V = pickle.load(f)
      with open('{}/SVD_2.pkl'.format(local_dir_name), 'r') as f:
        U1, D1, V1 = pickle.load(f)
      true_dims = range(rank)
    except:
      D_gen = D[:rank]
      U_gen = generate_random_orthogonal_matrix((n, rank)) 
      V_gen = generate_random_orthogonal_matrix((n, rank))
      assert(len(D_gen) == rank)
      true_dims = range(rank)

      X = (U_gen * D_gen).dot(V_gen.T)
      Y = np.array(X)

      E = np.random.normal(0, args.sigma, size = shape)
      estimation_noise_E = E 
      frobenius_noise = np.linalg.norm(E, 'fro')

      Y += estimation_noise_E

      U, D, V = np.linalg.svd(X)
      U1, D1, V1 = np.linalg.svd(Y)
      try:
        with open('{}/SVD_1.pkl'.format(local_dir_name), 'w') as f:
          pickle.dump([U, D, V], f)
        with open('{}/SVD_2.pkl'.format(local_dir_name), 'w') as f:
          pickle.dump([U1, D1, V1], f)
      except:
        pass

    frobenius_list_est_to_gt = []
    var_sv_ub = []
    var_subspace_ub = []
    var_sv_gt = []
    var_subspace_gt = []
    telescoping_ub = []
    telescoping_expansion = []
    gt_ub = []

    embed_gt = U[:,true_dims] * (D[true_dims] ** alpha)
    sim_gt = embed_gt.dot(embed_gt.T) 

    embed = U * (D ** alpha)
    embed_est = U1 * (D1 ** alpha)
    spectrum_T_inv_norms = []
    spectrum_norms_gt = []

    sim = None
    sim_est = None
    inner_prod_U = None
    inner_prod_U1 = None

    for keep_dim in range(1, rank):
      if sim is None:
        sim = embed[:,:keep_dim].dot(embed[:,:keep_dim].T)
        sim_est = embed_est[:,:keep_dim].dot(embed_est[:,:keep_dim].T)
      else:
        sim += np.outer(embed[:,keep_dim-1], embed[:,keep_dim-1])
        sim_est += np.outer(embed_est[:,keep_dim-1], embed_est[:,keep_dim-1])

      sim_diff_est_to_gt = np.linalg.norm(sim_est - sim_gt, 'fro')
      # now, the three error terms
      bias = np.sqrt(np.sum(np.power(D[keep_dim:rank], 4 * alpha)))
      var_sv_gt.append(np.sqrt(np.sum((np.power(D[:keep_dim], 2 * alpha) - np.power(D1[:keep_dim], 2 * alpha)) ** 2)))
      var_sv = 2 * alpha * np.sqrt(2*n) * sigma * np.sqrt(np.sum(
        np.power(D[:keep_dim], 4 * alpha - 2)))
      ''' use G.W. Stewart's Wietland-Hoffman theorem'''
      if alpha == 0.5:
        var_sv = keep_dim * sigma
      var_sv_ub.append(var_sv)
      
      coeffs = [np.sqrt(i * (rank-i)) for i in range(1, keep_dim+1)]
      var_subspace_sine = 2 * np.sqrt(2) * alpha * sigma * np.sum(np.power(D[:keep_dim], 2 * alpha - 1) * coeffs)
      spectrum_T_inv = 1 / (np.outer(D[:keep_dim], np.ones((n - keep_dim, 1))) - 
        np.outer(np.ones((keep_dim, 1)), D[keep_dim:]))
      spectrum_T_inv_norms.append(np.linalg.norm(spectrum_T_inv, 'fro'))
      if inner_prod_U is None:
        inner_prod_U = U[:,:keep_dim].dot(U[:,:keep_dim].T)
        inner_prod_U1 = U1[:,:keep_dim].dot(U1[:,:keep_dim].T)
      else:
        inner_prod_U += np.outer(U[:,keep_dim-1], U[:,keep_dim-1])    
        inner_prod_U1 += np.outer(U1[:,keep_dim-1], U1[:,keep_dim-1])    
      spectrum_norms_gt.append(np.linalg.norm(inner_prod_U - inner_prod_U1, 'fro') / np.sqrt(2))
      var_subspace_expansion = 0
      var_subspace_expansion_gt = 0
      for i in range(keep_dim):
        var_subspace_expansion += np.sqrt(2) * sigma * (
          (np.power(D[i], 2 * alpha) - np.power(D[i+1], 2 * alpha))
          * spectrum_T_inv_norms[i])
        var_subspace_expansion_gt += np.sqrt(2) * (
          (np.power(D[i], 2 * alpha) - np.power(D[i+1], 2 * alpha))
          * spectrum_norms_gt[i])
      var_subspace_ub.append(var_subspace_expansion)
      var_subspace_gt.append(var_subspace_expansion_gt)
      
      telescoping_ub.append(bias + var_subspace_sine + var_sv)
      telescoping_expansion.append(bias + var_subspace_expansion + var_sv)
      gt_ub.append(bias + var_subspace_expansion_gt + var_sv_gt[-1])
      frobenius_list_est_to_gt.append(sim_diff_est_to_gt) 
    print(frobenius_list_est_to_gt[-1])
    frobenius_list_est_to_gt_dict.append(frobenius_list_est_to_gt)
    var_sv_ub_dict.append(var_sv_ub)
    var_subspace_ub_dict.append(var_subspace_ub)
    var_sv_gt_dict.append(var_sv_gt)
    var_subspace_gt_dict.append(var_subspace_gt)
    telescoping_ub_dict.append(telescoping_ub)
    telescoping_expansion_dict.append(telescoping_expansion)
    gt_ub_dict.append(gt_ub)

    # telescoping_ub_min_x = np.argmin(telescoping_ub)
    # telescoping_ub_min_y = np.min(telescoping_ub)
    # gt_ub_min_x = np.argmin(gt_ub)
    # gt_ub_min_y = np.min(gt_ub)
    # telescoping_expansion_min_x = np.argmin(telescoping_expansion)
    # telescoping_expansion_min_y = np.min(telescoping_expansion)
    # gt_min_x = np.argmin(frobenius_list_est_to_gt)
    # gt_min_y = np.min(frobenius_list_est_to_gt)

  with open('{}/results.pkl'.format(dir_name), 'w') as f:
    pickle.dump([frobenius_list_est_to_gt_dict, var_sv_ub_dict,
      var_subspace_ub_dict,
      var_sv_gt_dict,
      var_subspace_gt_dict,
      telescoping_ub_dict,
      telescoping_expansion_dict,
      gt_ub_dict], f)



'''first plot: overall bound quality vs ground truth'''
fig = plt.figure()
ax  = fig.add_subplot(111)

for i in range(len(telescoping_expansion_dict)):
  frobenius_list_est_to_gt = frobenius_list_est_to_gt_dict[i]
  gt_min_x = np.argmin(frobenius_list_est_to_gt)
  gt_min_y = frobenius_list_est_to_gt[gt_min_x]
  # calculate in what interval does error stay within x% of optimal
  conf_int_ub = gt_min_y * (1 + conf_int)
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
  telescoping_expansion = telescoping_expansion_dict[i]
  gt_ub = gt_ub_dict[i]
  if i == 0:
    # ax.plot(telescoping_ub, 'yellow', label = r'$\sqrt{\sum_{i=k}^d \lambda_i^{4\alpha}} +\sqrt{2}\sum_{i=1}^k(\lambda_i^{2\alpha}-\lambda_{i+1}^{2\alpha})\|\hat U_{\cdot,1:i}^TU_{\cdot,i:n}\|+\sqrt{\sum_{i=1}^k (\lambda_i^{2\alpha}-\hat\lambda_{i}^{2\alpha})^2}$')
    # ax.plot(gt_ub, 'green', label = r'PIP Error by Theorem 2')
    ax.plot(frobenius_list_est_to_gt, 'aqua', label = r'Actual PIP Error')
    ax.plot(telescoping_expansion, 'red', label = r'PIP Error Bound by Main Theorem')
  else:
    # ax.plot(telescoping_ub, 'yellow')
    # ax.plot(gt_ub, 'green')
    ax.plot(frobenius_list_est_to_gt, 'aqua')
    # ax.plot(telescoping_expansion, 'red')
  # ax.annotate('{:.2f} at {}'.format(gt_min_y, gt_min_x),
      # xy = (gt_min_x, gt_min_y), xytext = (gt_min_x-10, gt_min_y - 5),
      # arrowprops=dict(facecolor='black', shrink=0.05),)
  # ax.annotate('{:.2f} at {}'.format(gt_ub_min_y, gt_ub_min_x),
      # xy = (gt_ub_min_x, gt_ub_min_y), xytext = (gt_ub_min_x-10, gt_ub_min_y - 5),
      # arrowprops=dict(facecolor='black', shrink=0.05),)
  # ax.annotate('{:.2f} at {}'.format(telescoping_ub_min_y, telescoping_ub_min_x),
      # xy = (telescoping_ub_min_x, telescoping_ub_min_y), xytext = (telescoping_ub_min_x-10, telescoping_ub_min_y - 5),
      # arrowprops=dict(facecolor='black', shrink=0.05),)
  # ax.annotate('{:.2f} at {}'.format(telescoping_expansion_min_y, telescoping_expansion_min_x),
      # xy = (telescoping_expansion_min_x, telescoping_expansion_min_y), xytext = (telescoping_expansion_min_x-10, telescoping_expansion_min_y - 5),
      # arrowprops=dict(facecolor='black', shrink=0.05),)
lgd = ax.legend(loc='upper left')
plt.title(r'Comparing PIP Error Theorem Predictions, $\alpha$={}'.format(alpha))
plt.xlabel('Dimensions')
plt.ylabel('PIP Error')
fig.savefig('{}/bound_{}.pdf'.format(dir_name, alpha), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

'''second plot: bound quality on singular values vs ground truth'''
fig = plt.figure()
ax  = fig.add_subplot(111)
for i in range(len(telescoping_expansion_dict)):
  var_sv_ub = var_sv_ub_dict[i]
  var_sv_gt = var_sv_gt_dict[i]
  if i == 0:
    ax.plot(var_sv_ub, 'aqua', label = r'upper bound on singular values')
    ax.plot(var_sv_gt, 'red', label = r'ground truth on singular values')
  else:
    ax.plot(var_sv_ub, 'aqua')
    ax.plot(var_sv_gt)
lgd = ax.legend(loc='upper right')
plt.title(r'sv perturbation'.format(alpha))
fig.savefig('{}/sval_{}.pdf'.format(dir_name, alpha), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

'''third plot: bound quality on singular vectors vs ground truth'''
fig = plt.figure()
ax  = fig.add_subplot(111)
for i in range(len(telescoping_expansion_dict)):
  var_subspace_ub = var_subspace_ub_dict[i]
  var_subspace_gt = var_subspace_gt_dict[i]
  if i == 0:
    ax.plot(var_subspace_ub, 'aqua', label = r'upper bound on singular vectors')
    ax.plot(var_subspace_gt, 'red', label = r'ground truth on singular vectors')
  else:
    ax.plot(var_subspace_ub, 'aqua')
    ax.plot(var_subspace_gt) 
lgd = ax.legend(loc='upper right')
plt.title(r'sv perturbation'.format(alpha))
fig.savefig('{}/svec_{}.pdf'.format(dir_name, alpha), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

'''fourth plot: ground truth'''
fig = plt.figure()
ax  = fig.add_subplot(111)
for i in range(len(telescoping_expansion_dict)):
  frobenius_list_est_to_gt = frobenius_list_est_to_gt_dict[i]
  if i == 0:
    ax.plot(frobenius_list_est_to_gt, 'aqua', label = r'grount truth PIP error')
  else:
    ax.plot(frobenius_list_est_to_gt, 'aqua')
lgd = ax.legend(loc='upper right')
plt.title(r'PIP Error Ground Truth'.format(alpha))
fig.savefig('{}/pip_gt_{}.pdf'.format(dir_name, alpha), bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()