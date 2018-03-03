import cPickle as pickle
import numpy as np
import os.path
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_with_labels(low_dim_embs, labels, filename='2_dim.pdf'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 fontsize = 30,
                 ha='right',
                 va='bottom')
  plt.show()
  plt.savefig(filename)
  plt.close()

path = './l2v_params/'
if_exists = False

if not if_exists:
  with open(path + 'count.pkl', 'r') as f:
    dictionary = pickle.load(f)
  with open(path + 'embeddings_E.pkl', 'r') as f:
    unnormalized_embeddings_ = pickle.load(f)
  with open(path + 'nce_embeddings.pkl', 'r') as f:
    nce_embeddings_ = pickle.load(f)
  with open(path + 'nce_biases.pkl', 'r') as f:
    nce_biases_ = pickle.load(f)

  print unnormalized_embeddings_.shape
  U, D, V = np.linalg.svd(unnormalized_embeddings_)
  U_2 = U[:,:2]
  print U_2.shape
  with open(path + 'reduced_dim.pkl', 'w') as f:
    pickle.dump(U_2, f)

with open(path + 'reduced_dim.pkl', 'r') as f:
  U_2 = pickle.load(f)
x = U_2[:,0]
y = U_2[:,1]
z = [len(x) - i for i in range(len(x))]

plot_with_labels(U_2, dictionary.keys())

# plt.scatter(x, y)
# plt.savefig('2_dim.pdf')
# plt.close()

for d in D:
  print d

plt.plot(D)
plt.savefig('sv.pdf')
plt.close()

xi = np.linspace(x.min(),x.max(),1000)
yi = np.linspace(y.min(),y.max(),1000)
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
zmax = np.max(z)
zmin = np.min(z)

CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow, vmax=zmax, vmin=zmin)
plt.colorbar() 
plt.savefig('heatmap.pdf')







