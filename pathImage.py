"""Crop a polygonal selection from an image."""
import numpy as np
from PIL import Image
from shapely.geometry import Point
from shapely.geometry import Polygon
import os
import logging

def folderRecursive(mainFolder, target, coor):
    cont = 0
    for root, subdirs, files in os.walk(mainFolder):
        for file in os.listdir(root):
            filePath = os.path.join(root, file)
            if os.path.isdir(filePath):
                pass
            else:
                cont += 1
                logging.info(str(target) + " " + str(cont) + "- " + filePath)
                # print filePath
                crop(filePath, target + filePath.split('/')[1] + "/" + file, coor)


def crop(image_path, saved_location, coords):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    # cropped_image.show()

def count_files():
    count = 1
    for root, subdirs, files in os.walk("train_croped"):
        for file in os.listdir(root):
            filePath = os.path.join(root, file)
            if os.path.isdir(filePath):
                pass
            else:
                count += 1

    return count

logging.basicConfig(filename='log', level=logging.DEBUG)
folderRecursive("train", "train_croped_1024_1024_512/", (200,200,512,512))
folderRecursive("train", "train_croped_1536_1546_512/", (400,400,700,700))
folderRecursive("train", "train_croped_2048_2048_512/", (600,600,900,900))