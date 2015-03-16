import graphlab as gl
import re
import random
from copy import copy
import os

# Run this script in the same directory as the train/ test/ and
# processed/ directories -- where you ran the prep_image.sh.  It will
# put a image-sframes/ directory with train and test SFrames in the
# save_path location below. 

save_path = "./"

# gl.set_runtime_config("GRAPHLAB_CACHE_FILE_LOCATIONS", os.path.expanduser("~/data/tmp/"))

# shuffle the training images
X = gl.image_analysis.load_images("processed/")
X["is_train"] = X["path"].apply(lambda p: "train" in p)

# Add in all the relevant information in places
source_f = lambda p: re.search("run-(?P<source>[^/]+)", p).group("source")
X["source"] = X["path"].apply(source_f)

extract_name = lambda p: re.search("[0-9]+_(right|left)", p).group(0)
X["name"] = X["path"].apply(extract_name)

X_train = X[X["is_train"] == True]
X_test = X[X["is_train"] != True]

# Add in the training labels
labels_sf = gl.SFrame.read_csv("trainLabels.csv")
label_d = dict( (d["image"], d["level"]) for d in labels_sf)

X_train["level"] = X_train["name"].apply(lambda p: label_d[p])

# Get roughly equal class representation by duplicating the different levels.
X_train_levels = [X_train[X_train["level"] == lvl] for lvl in [1,2,3,4] ]
n_dups = [int(round((1.0/5) / (float(xtl.num_rows()) / X_train.num_rows()) )) for xtl in X_train_levels]

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
