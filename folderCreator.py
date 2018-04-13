import os
import sys

for root, subdirs, files in os.walk("train"):

    print subdirs
    new_dir = "train_croped/"

    for dir in subdirs:
        if not os.path.exists(new_dir + dir):
            os.makedirs(new_dir + dir)
        else:
            print dir + " ja existe"