import graphlab as gl
import re
import random
from copy import copy
import os
from os.path import join, abspath
from itertools import chain

random.seed(0)

# Run this script in the same directory as the train/ test/ and
# processed/ directories -- where you ran the prep_image.sh.  It will
# put a image-sframes/ directory with train and test SFrames in the
# save_path location below. 

save_path = "./"
image_path = "/data/hoytak/diabetic-retinopathy-code/data"

# Don't load the images yet.  This is done later, with a random
# transformation.

X = gl.SFrame()

# Load all images in the 
X["path"] = list(chain(*[[abspath(join(root, f)) for f in files
                          if f.endswith('jpeg')]
                         for root, dir_list, files in os.walk(image_path)]))

# shuffle the training images
X = gl.image_analysis.load_images("processed/")
X["is_train"] = X["path"].apply(lambda p: "train" in p)

# Add in all the relevant information in places
source_f = lambda p: re.search("run-(?P<source>[^/]+)", p).group("source")
X["source"] = X["path"].apply(source_f)

extract_number = lambda p: re.search("([0-9]+)_(right|left)", p).group(0)
X["number"] = X["path"].apply(extract_number)

extract_name = lambda p: re.search("([0-9]+)_(right|left)", p).group(1)
X["name"] = X["path"].apply(extract_name)

X_train = X[X["is_train"] == True]
X_test = X[X["is_train"] != True]

# Add in the training labels
labels_sf = gl.SFrame.read_csv("trainLabels.csv")
label_d = dict( (d["image"], d["level"]) for d in labels_sf)

X_train["level"] = X_train["name"].apply(lambda p: label_d[p])

# Get roughly equal class representation by duplicating the different
# levels.  0 is the most well represented class, so use it as reference.

def shuffle(Xt):
    # Sort the rows
    Xt["random"] = np.random.uniform(size = Xt.num_rows())
    Xt = Xt.sort("random")
    del Xt["random"]

def balance_classes(Xt):

    masks = [(Xt["level"] == lvl) for lvl in range(5)]

    target_n = max(mask.sum() for mask in masks)

    X_new = copy(Xt)

    for lvl, mask in enumerate(masks):
        n_dups = (float(target_n) / mask.sum()) - 1

        X_add = X_train[mask]
        shuffle(X_add)

        while n_dups > 1:
            X_new = X_new.append(X_add)
            n_dups -= 1

        n_samples = int(n_dups * X_add.num_rows())
        
        if n_samples > 0:
            X_new.append(X_add[:n_samples])
            
        
def load_perturbed_image(d):

    in_file = d['path']
    seed = d['row_number']

    out_file = join(out_file_base, image_path)
    
    subprocess.call('./make_perturbed_image.sh %s %s %d' % (in_file, out_file, seed), shell=True)
    
    return gl.Image(out_file)
    




    
    
    
        


X_train_levels = [] for lvl in 
n_dups = [((1.0/5) / (float(xtl.num_rows()) / X_train.num_rows()) )) for xtl in X_train_levels]

for nd, xtl_src in zip(n_dups, X_train_levels):
    for i in range(nd):
        X_train = X_train.append(xtl_src)
        
# Do a poor man's random shuffle
X_train["_random_"] = random.sample(xrange(X_train.num_rows()), X_train.num_rows())
X_train = X_train.sort("_random_")
del X_train["_random_"]

# Save sframes to a bucket
X_train.save(save_path + "image-sframes/train")
X_test.save(save_path + "image-sframes/test")
