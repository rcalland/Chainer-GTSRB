from __future__ import print_function

import random
import argparse
import cv2
import os
import matplotlib
import collections
import fnmatch
import operator
import math
from utils import load_config

def make_augmentations(img_path, num_copies):
    # taken from http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
   
    pixel_jitter = 4
    rotate_degrees = 25
    scale_min = 0.8
    scale_max = 1.2

    img = cv2.imread(img_path)
    for i in range(num_copies):
        scale_factor = random.uniform(scale_min, scale_max)
        rotate_factor = random.uniform(-rotate_degrees, rotate_degrees)
        transx_factor = random.randint(-pixel_jitter, pixel_jitter)
        transy_factor = random.randint(-pixel_jitter, pixel_jitter)
        rows,cols,ch = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), rotate_factor, scale_factor)
        M[0][2] += transx_factor
        M[1][2] += transy_factor
        dst = cv2.warpAffine(img,M,(cols,rows))
        
        # save this image
        newname = "{}_oversampled_{}.ppm".format(img_path, i)
        cv2.imwrite(newname, dst)

def get_class(path):
    # get the last directory name
    class_label = os.path.basename(os.path.normpath(path))
    # strip off zero padding
    class_label = class_label.lstrip("0")
    if class_label == "":
        class_label = "0"
    return class_label

def fetch_list(base_dir):
    print("collecting images...")
    base_dir = os.path.join(base_dir, "GTSRB/Final_Training/Images")

    sample_list = []
    label_list = []
    print(base_dir)

    #for root, dirs, files in os.walk(base_dir, topdown=False):
    for dirs in os.listdir(base_dir):
        class_path = os.path.join(base_dir, dirs)
        #print(class_path)
        for f in os.listdir(class_path):
            #print(f)
            full_path = os.path.join(class_path, f)
            #print(full_path)
        
            # remove previously, generated oversamples
            if fnmatch.fnmatch(f, "*oversampled*"):
                print("removing {}".format(full_path))
                os.remove(full_path)

            elif f.endswith(".ppm"):
                lbl = get_class(class_path)
                sample_list.append(full_path)
                label_list.append(lbl)

    print("done.")
    return sample_list, label_list

def class_frequency(labels):
    return collections.OrderedDict(collections.Counter(labels))

def upsample(freq, sample_list, label_list, global_scale=0):
    print("beginning upsampling...")
    # get the class with the most samples
    maxkey = max(freq.iteritems(), key=operator.itemgetter(1))[0]
    maxsamples = freq[maxkey]
    #print(maxkey, maxsamples)

    total_files = len(sample_list)

    # perform sampling
    for i, (spl, lbl) in enumerate(zip(sample_list, label_list)):
        if i % 1000 is 0:
            print("{} / {}".format(i, total_files))
        num_augmentations = maxsamples - freq[lbl]
        num_augmentations = int(math.ceil(num_augmentations / float(freq[lbl]))) + int(maxsamples * global_scale)
        make_augmentations(spl, num_augmentations)
    print("done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", "-b", type=str, default=None)
    #parser.add_argument("--balanced", type=bool, default=False)
    args = parser.parse_args()

    if args.base_dir is None:
        cfg = load_config("../config/gtsrb.json")
        args.base_dir = cfg["data_root_path"]

    sample_list, label_list = fetch_list(args.base_dir)
    freq = class_frequency(label_list)
    
    # global_scale increases all class samples by %
    upsample(freq, sample_list, label_list, 0.0)

if __name__=="__main__":
    main()
