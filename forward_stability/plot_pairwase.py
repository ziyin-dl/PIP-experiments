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
import seaborn as sns; sns.set()
import pandas as pd

if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", nargs='+', required=True, type=str)
  args = parser.parse_args()
  key0 = args.folder[0]
  with open(os.path.join(key0, "pip_rel_loss.npx"), 'r') as f:
    cross_check_matrix = np.load(f)
    # cross_check_matrix = 1.0 / cross_check_matrix
  with open(os.path.join(key0, "pip_abs_loss.npx"), 'r') as f:
    loss_matrix = np.load(f)
  # 1 to 100: increment of 1; 100 to 145: increment of 20; 145-155: increment of 1000
  dimensions = [range(0,100), range(99, 145), range(144, 154)]
  real_dimensions = [range(1,101), [i*20 for i in range(5, 51)], [i*1000 for i in range(1, 11)]]
  for idx, dimension in enumerate(dimensions):
    ixgrid = np.ix_(dimension, dimension)
    real_dimension = real_dimensions[idx]
    plt.figure()
    df = pd.DataFrame(cross_check_matrix[ixgrid], index=real_dimension, columns=real_dimension)
    ax = sns.heatmap(df, xticklabels=len(dimension)-1, yticklabels=len(dimension)-1)
    # ax = sns.heatmap(df, xticklabels=len(dimension)-1, yticklabels=len(dimension)-1, vmin=0, vmax=1)
    fig = ax.get_figure()
    fig.savefig("PIP_loss_normalized_{}.pdf".format(idx))
    
    plt.figure()
    df = pd.DataFrame(loss_matrix[ixgrid], index=real_dimension, columns=real_dimension)
    ax = sns.heatmap(df, xticklabels=len(dimension)-1, yticklabels=len(dimension)-1)
    # ax = sns.heatmap(df, xticklabels=len(dimension)-1, yticklabels=len(dimension)-1, vmin=0, vmax=1)
    fig = ax.get_figure()
    fig.savefig("PIP_loss_{}.pdf".format(idx))
    
