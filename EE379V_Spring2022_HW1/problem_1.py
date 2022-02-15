import cv2
import matplotlib.pyplot as plt
import numpy as np
from tight_subplot import tight_subplot_2x5

img = cv2.imread('texture.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = cv2.getGaussianKernel(3, 1)
g_pyr = [img]
l_pyr = []
for i in range(1, 5):
  # apply gaussian filter
  gaussed_img = cv2.filter2D(g_pyr[i-1], -1, kernel)
  g_img_arr = np.array(gaussed_img)
  ungaussed_img_arr = np.array(g_pyr[i-1])
  # save residual i.e. diff of pre-and-post gauss blur img
  residual = np.subtract(ungaussed_img_arr, g_img_arr)
  l_pyr.append(residual)
  # remove even rows and columns
  rows = g_img_arr.shape[0]
  cols = g_img_arr.shape[1]
  g_img_arr = np.delete(g_img_arr, list(range(0, rows, 2)), axis=0)
  g_img_arr = np.delete(g_img_arr, list(range(0, cols, 2)), axis=1)
  # convert back to cv2 greyscale image (unecessary but consistent)
  downsampled_img = cv2.cvtColor(g_img_arr, cv2.COLOR_GRAY2BGR)
  g_pyr.append(g_img_arr)

tight_subplot_2x5(g_pyr, l_pyr)
plt.savefig('plot1.1.png')

  # np.fft.fftshift()