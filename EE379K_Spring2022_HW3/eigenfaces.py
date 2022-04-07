import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import cv2
import sklearn
from readFaceImages import readFaceImages

# first d eigenvectors
D = 9

images, people, numbers, subsets = readFaceImages("./faces/")
im_vects = []

### Part 1

for im in images:
  im = np.array(im)
  # flatten
  im_vect = im.flatten()
  # SVD
  im_vects.append(im_vect)
im_vects = np.array(im_vects)
#normalize by subtracting mean of image vectors
im_vects = im_vects - np.mean(im_vects, axis=0)

u, s, vh = la.svd(im_vects)
vh = np.array(vh)
d_vectors = []
for i in range(D):
  d_vectors.append(np.reshape(vh[i], (50, 50)))
d_vectors = np.array(d_vectors)

# Show eigenface images
for i in range(D):
  img = plt.imshow(d_vectors[i])
  img.set_cmap('gray')
  plt.show()

### Part 2
subs = [0, 7, 19, 31, 45, 64]
