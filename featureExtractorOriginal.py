#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
import PIL
from PIL import Image


# Load image
original = pywt.data.aero()
# Load image
img = PIL.Image.open("train_croped/iPhone-4s/(iP4s)40.jpg").convert("RGB")
imgarr = np.array(img)

# im = Image.open(imgpath)
# pixels = list(im.getdata())

b, g, r = imgarr[:, :, 0], imgarr[:, :, 1], imgarr[:, :, 2]  # For RGB image

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(b, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)

plt.show()