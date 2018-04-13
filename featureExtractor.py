#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from array import array
from scipy.stats import skew
from scipy.stats import kurtosis

import pywt
import pywt.data
import csv
import logging


def higherOrderWaveletFeaturesExtractor(imgarr, filename, filepath, writer):
    logging.basicConfig(filename='log', level=logging.DEBUG)

    cont = 0
    for rgb in imgarr:
        if cont == 0: mode = "r"
        if cont == 1: mode = "g"
        if cont == 2: mode = "b"
        cont += 1

        original = rgb

        coeffs2 = pywt.dwt2(original, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2

        # titles = ['Horizontal detail', 'Vertical detail', 'Diagonal detail']

        if  mode == "r":
            r_horizontal_mean = np.mean(LH)
            r_horizontal_variance = np.var(LH)
            r_horizontal_skewness = skew(LH.flat)
            r_horizontal_kurtosis = kurtosis(LH.flat)

            r_vertical_mean = np.mean(HL)
            r_vertical_variance = np.var(HL)
            r_vertical_skewness = skew(HL.flat)
            r_vertical_kurtosis = kurtosis(HL.flat)

            r_diagonal_mean = np.mean(HH)
            r_diagonal_variance = np.var(HH)
            r_diagonal_skewness = skew(HH.flat)
            r_diagonal_kurtosis = kurtosis(HH.flat)

            logging.info(" - R")


        if mode == "g":
            g_horizontal_mean = np.mean(LH)
            g_horizontal_variance = np.var(LH)
            g_horizontal_skewness = skew(LH.flat)
            g_horizontal_kurtosis = kurtosis(LH.flat)

            g_vertical_mean = np.mean(HL)
            g_vertical_variance = np.var(HL)
            g_vertical_skewness = skew(HL.flat)
            g_vertical_kurtosis = kurtosis(HL.flat)

            g_diagonal_mean = np.mean(HH)
            g_diagonal_variance = np.var(HH)
            g_diagonal_skewness = skew(HH.flat)
            g_diagonal_kurtosis = kurtosis(HH.flat)

            logging.info(" - G")


        if mode == "b":
            b_horizontal_mean = np.mean(LH)
            b_horizontal_variance = np.var(LH)
            b_horizontal_skewness = skew(LH.flat)
            b_horizontal_kurtosis = kurtosis(LH.flat)

            b_vertical_mean = np.mean(HL)
            b_vertical_variance = np.var(HL)
            b_vertical_skewness = skew(HL.flat)
            b_vertical_kurtosis = kurtosis(HL.flat)

            b_diagonal_mean = np.mean(HH)
            b_diagonal_variance = np.var(HH)
            b_diagonal_skewness = skew(HH.flat)
            b_diagonal_kurtosis = kurtosis(HH.flat)

            logging.info(" - B")

    logging.info("Done...")
    logging.info("")
    result = {
        "filename": filename,
        "fullpath": filepath,


        "r_horizontal_mean": r_horizontal_mean,
        "r_horizontal_variance": r_horizontal_variance,
        "r_horizontal_skewness": r_horizontal_skewness,
        "r_horizontal_kurtosis": r_horizontal_kurtosis,

        "g_horizontal_mean": g_horizontal_mean,
        "g_horizontal_variance": g_horizontal_variance,
        "g_horizontal_skewness": g_horizontal_skewness,
        "g_horizontal_kurtosis": g_horizontal_kurtosis,

        "b_horizontal_mean": b_horizontal_mean,
        "b_horizontal_variance": b_horizontal_variance,
        "b_horizontal_skewness": b_horizontal_skewness,
        "b_horizontal_kurtosis": b_horizontal_kurtosis,


        "r_vertical_mean": r_vertical_mean,
        "r_vertical_variance": r_vertical_variance,
        "r_vertical_skewness": r_vertical_skewness,
        "r_vertical_kurtosis": r_vertical_kurtosis,

        "g_vertical_mean": g_vertical_mean,
        "g_vertical_variance": g_vertical_variance,
        "g_vertical_skewness": g_vertical_skewness,
        "g_vertical_kurtosis": g_vertical_kurtosis,

        "b_vertical_mean": b_vertical_mean,
        "b_vertical_variance": b_vertical_variance,
        "b_vertical_skewness": b_vertical_skewness,
        "b_vertical_kurtosis": b_vertical_kurtosis,


        "r_diagonal_mean": r_diagonal_mean,
        "r_diagonal_variance": r_diagonal_variance,
        "r_diagonal_skewness": r_diagonal_skewness,
        "r_diagonal_kurtosis": r_diagonal_kurtosis,

        "g_diagonal_mean": g_diagonal_mean,
        "g_diagonal_variance": g_diagonal_variance,
        "g_diagonal_skewness": g_diagonal_skewness,
        "g_diagonal_kurtosis": g_diagonal_kurtosis,

        "b_diagonal_mean": b_diagonal_mean,
        "b_diagonal_variance": b_diagonal_variance,
        "b_diagonal_skewness": b_diagonal_skewness,
        "b_diagonal_kurtosis": b_diagonal_kurtosis
    }

    writer.writerow(result)


def convertImgToArray(imgpath):
    # Load image
    img = PIL.Image.open(imgpath).convert("RGB")
    imgarr = np.array(img)

    # im = Image.open(imgpath)
    # pixels = list(im.getdata())

    b, g, r = imgarr[:, :, 0], imgarr[:, :, 1], imgarr[:, :, 2]  # For RGB image

    return [
        r,
        g,
        b
    ]