import os
import sys

for root, subdirs, files in os.walk("train"):

    print subdirs
    new_dir = "train_croped_2048_2048_512/"

    for dir in subdirs:
        if not os.path.exists(new_dir + dir):
            os.makedirs(new_dir + dir)
        else:
            print dir + " ja existe"

