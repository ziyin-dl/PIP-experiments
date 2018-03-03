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

if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  with open("crossval.pkl", 'r') as f:
    dims, losses, nsrs = np.load(f)

  plt.plot(dims, nsrs)
  plt.title("PIP Loss Between Two Runs")
  plt.xlabel("Dimension")
  plt.ylabel("PIP Loss Noise To Signal Ratio")
  plt.savefig("PIP_nsr_vs_dim.pdf")
  plt.close()
  plt.plot(dims, losses)
  plt.title("PIP Loss Between Two Runs")
  plt.xlabel("Dimension")
  plt.ylabel("PIP Loss")
  plt.savefig("PIP_loss_vs_dim.pdf")
  plt.close()

