import graphlab as gl
import re
import random
from copy import copy
import os
from os.path import join, abspath, expanduser, split
from itertools import chain
import subprocess

random_seed = 0
n_duplicates = 4

random.seed(random_seed)

# Run this script in the same directory as the train/ test/ and
# processed/ directories -- where you ran the prep_image.sh.  It will
# put a image-sframes/ directory with train and test SFrames in the
# save_path location below. 

base_path     = expanduser("~/data/diabetic/")
save_path     = join(base_path, "data-sframes/")
data_path     = join(base_path, "data/")
out_file_base = join(base_path, "processed-data/")


# Don't load the images yet.  This is done later, with a random
# transformation.

X = gl.SFrame()

# Load all images in the 
X["path"] = list(chain(*[[abspath(join(root, f)) for f in files
                          if f.endswith('jpeg')]
                         for root, dir_list, files in os.walk(data_path)]))

# shuffle the training images
X["is_train"] = X["path"].apply(lambda p: "train" in p)

extract_number = lambda p: re.search("([0-9]+)_(right|left)", p).group(1)
X["number"] = X["path"].apply(extract_number)

extract_name = lambda p: re.search("([0-9]+)_(right|left)", p).group(0)
X["name"] = X["path"].apply(extract_name)

X_data = X[X["is_train"] == True]
X_test = X[X["is_train"] != True]

# Add in the training labels
labels_sf = gl.SFrame.read_csv(join(data_path, "trainLabels.csv"))
label_d = dict( (d["image"], d["level"]) for d in labels_sf)

X_data["level"] = X_data["name"].apply(lambda p: label_d[p])

# Get roughly equal class representation by duplicating the different
# levels.  0 is the most well represented class, so use it as reference.

def shuffle(Xt):
    # Sort the rows
    Xt["random"] = [random.random() for i in xrange(Xt.num_rows())]
    Xt = Xt.sort("random")
    del Xt["random"]
    return Xt

def balance_classes(Xt):

    masks = [(Xt["level"] == lvl) for lvl in range(5)]

    target_n = max(mask.sum() for mask in masks)

    X_new = copy(Xt)

    for lvl, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        
        n_dups = (float(target_n) / mask.sum()) - 1

        X_add = shuffle(Xt[mask])
        
        while n_dups > 1:
            X_new = X_new.append(X_add)
            n_dups -= 1

        n_samples = int(n_dups * X_add.num_rows())
        
        if n_samples > 0:
            X_new.append(X_add[:n_samples])

    return shuffle(X_new)

def attempt_image_load(in_file, out_file):
    try:
        return gl.Image(out_file)
    except Exception:
        pass

    for fuzz in [8, 6, 4, 2, 1]:
        try:
            subprocess.check_call('./make_conservative_image.sh %s %s %d' % (in_file, out_file, fuzz), shell=True)
            return gl.Image(out_file)
        except Exception:
            pass
        
    return None
    
        
def load_perturbed_image(d):

    in_file = d['path']
    seed = "%d-%d" % (random_seed, d['row_number'])

    out_file = (out_file_base + in_file).replace(".jpeg", "-perturbed-%s.jpeg" % seed)

    base_dir = split(out_file)[0]
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(out_file):
        try:
            subprocess.check_call('./make_perturbed_image.sh %s %s %s' % (in_file, out_file, seed), shell=True)
        except Exception:
            pass

    return attempt_image_load(in_file, out_file)

        
def load_normal_image(in_file):

    out_file = out_file_base + in_file

    base_dir = split(out_file)[0]
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(out_file):
        subprocess.check_call('./make_normal_image.sh %s %s' % (in_file, out_file), shell=True)

    return attempt_image_load(in_file, out_file)
            

def get_level(X_data, level):
    Xd = copy(X_data)
    Xd["level"] = (X_data["level"] >= level)
    return Xd


X_data_levels = [X_data] + [get_level(X_data, i) for i in [1,2,3,4]]

for i, Xd_src in enumerate(X_data_levels):
    train_save_path = join(save_path, "image-sframes/train-%d-%d" % (i, random_seed))
    train_raw_save_path = join(save_path, "image-sframes/train-raw-%d-%d" % (i, random_seed))
    valid_save_path = join(save_path, "image-sframes/valid-%d-%d" % (i, random_seed))

    running_train = not os.path.exists(train_save_path)
    running_train_raw = not os.path.exists(train_raw_save_path)
    running_valid = not os.path.exists(valid_save_path)
    
    if running_train or running_valid or running_train_raw:
        Xt_src, Xv = Xd_src.random_split(0.95, 1)
        
    if running_train:
        print "Generating set %d" % i

        Xt = copy(Xt_src)

        # Duplicate as needed.
        for j in range(n_duplicates - 1):
            Xt = Xt.append(Xt_src)

        # balance the training classes
        Xt = balance_classes(Xt)

        print "Loading training images"
        Xt = Xt.add_row_number("row_number")
        Xt["image"] = Xt[["path", "row_number"]].apply(load_perturbed_image)
        del Xt["row_number"]

        # # Save sframes to a bucket
        Xt = Xt.dropna()
        Xt.save(train_save_path)

    if running_valid:
        print "Loading validation images"
        Xv["image"] = Xv["path"].apply(load_normal_image)

        Xv = Xv.dropna()
        Xv.save(valid_save_path)

    if running_train_raw:
        print "Loading raw training images images"
        Xt = copy(Xt_src)
        Xt["image"] = Xt["path"].apply(load_normal_image)

        Xt.save(train_raw_save_path)
                

print "Loading images for test"

X_test["image"] = X_test["path"].apply(load_normal_image)
X_test.save(join(save_path, "image-sframes/test"))

X_train_raw = copy(X_data)
X_train_raw["image"] = X_train_raw["path"].apply(load_normal_image)
X_train_raw.save(join(save_path, "image-sframes/train-raw"))
