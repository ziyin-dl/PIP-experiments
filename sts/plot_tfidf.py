import cPickle as pickle
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('tfidf_sim_accuracy.pkl', 'r') as f:
 tf_idf_top = pickle.load(f)
print(tf_idf_top)
argmax = np.nanargmax(tf_idf_top)
print('max is {} at {}'.format(tf_idf_top[argmax], argmax))
