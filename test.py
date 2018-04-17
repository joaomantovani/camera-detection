from PIL import Image
import os
import time
import pywt
from featureExtractor import *
import sys
import csv
import logging


def count_files():
    count = 0
    for root, subdirs, files in os.walk("train_croped"):
        for file in os.listdir(root):
            filePath = os.path.join(root, file)
            if os.path.isdir(filePath):
                pass
            else:
                count += 1

    return count

def folderRecursive(mainFolder, csv_name):
    with open(csv_name, 'w') as csvfile:
        fieldnames = ["filename", "fullpath", "r_vertical_mean", "r_vertical_variance", "r_vertical_skewness",
                      "r_vertical_kurtosis", "g_vertical_mean", "g_vertical_variance", "g_vertical_skewness",
                      "g_vertical_kurtosis", "b_vertical_mean", "b_vertical_variance", "b_vertical_skewness",
                      "b_vertical_kurtosis", "r_horizontal_mean", "r_horizontal_variance", "r_horizontal_skewness",
                      "r_horizontal_kurtosis", "g_horizontal_mean", "g_horizontal_variance", "g_horizontal_skewness",
                      "g_horizontal_kurtosis", "b_horizontal_mean", "b_horizontal_variance", "b_horizontal_skewness",
                      "b_horizontal_kurtosis", "r_diagonal_mean", "r_diagonal_variance", "r_diagonal_skewness",
                      "r_diagonal_kurtosis", "g_diagonal_mean", "g_diagonal_variance", "g_diagonal_skewness",
                      "g_diagonal_kurtosis", "b_diagonal_mean", "b_diagonal_variance", "b_diagonal_skewness",
                      "b_diagonal_kurtosis"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print csv_name

        cont = 0
        for root, subdirs, files in os.walk(mainFolder):
            for file in os.listdir(root):
                filePath = os.path.join(root, file)
                if os.path.isdir(filePath):
                    pass
                else:
                    cont += 1
                    logging.info(str(cont) + "/" + str(count_files()) + " - " + filePath)
                    logging.info(str(cont * 100 / count_files()) + "%")
                    higherOrderWaveletFeaturesExtractor(convertImgToArray(filePath), filename=file, filepath=filePath, writer=writer)


start_time = time.clock()

logging.basicConfig(filename='log',level=logging.DEBUG)


# Comeca o feature extractor
folderRecursive("train_croped", 'features_croped_1024_1024_512.csv')
folderRecursive("train_croped", 'features_croped_1036_1036_512.csv')
folderRecursive("train_croped", 'features_croped_2048_2048_512.csv')

print count_files()

print ''
print time.clock() - start_time, "seconds"