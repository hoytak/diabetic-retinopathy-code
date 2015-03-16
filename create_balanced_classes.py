import graphlab as gl
import re
import random
from copy import copy
import os
import graphlab.aggregate as agg
import array
import numpy as np
import sys

# Run this script in the same directory as the 

train_path = "image-sframes/train-%d/"
valid_path = "image-sframes/validation-%d/"

X_data = gl.SFrame("image-sframes/train/")

def save_as_train_and_test(X, train_loc, valid_loc):

    # Can't just randomly sample the indices
    all_names = list(X["name"].unique())

    n_valid = (2 * len(all_names)) / 100
    
    random.shuffle(all_names)

    tr_names = gl.SArray(all_names[n_valid:])
    valid_names = gl.SArray(all_names[:n_valid])

    X_train = X.filter_by(tr_names, 'name')
    X_valid = X.filter_by(valid_names, 'name')

    X_train.save(train_loc)
    X_valid.save(valid_loc)

# The classes were already balanced by create_image_sframe, so we
# don't need to balance them below.
if not (os.path.exists(train_path % 0) and os.path.exists(valid_path % 0)):
    print "Skipping class 0; already present.  If error, remove these directories and restart."
    save_as_train_and_test(X_data, train_path % 0, valid_path % 0)

################################################################################
# Now do the other splitting parts

for mi in [1,2,3,4]:

    if os.path.exists(train_path % mi) and os.path.exists(valid_path % mi):
        print "Skipping class %d; already present.  If error, remove these directories and restart." % mi
        continue

    print "Running class %d" % mi
    
    X_data["class"] = (X_data["level"] >= mi)

    X_data_local = copy(X_data)

    n_class_0 = (X_data["class"] == 0).sum()
    n_class_1 = (X_data["class"] == 1).sum()

    if n_class_0 < n_class_1:

        num_to_sample = n_class_1 - n_class_0

        # Oversample the ones on the border
        level_to_sample = mi - 1
        class_to_sample = 0
        
    else:

        num_to_sample = n_class_0 - n_class_1

        # Oversample the ones on the border
        level_to_sample = mi
        class_to_sample = 1

    X_data_lvl = X_data[X_data["level"] == level_to_sample]

    # Do one extra of the closest class to slightly oversample the hard examples. 
    n = min(X_data_lvl.num_rows(), num_to_sample)
    X_data_local = X_data_local.append(X_data_lvl[:n])
    num_to_sample -= n

    if num_to_sample > 0:

        X_data_class = X_data[X_data["class"] == class_to_sample]
        
        while num_to_sample > 0:
            n = min(X_data_class.num_rows(), num_to_sample)
            X_data_local = X_data_local.append(X_data_class[:n])
            num_to_sample -= n

    # Sort the rows
    X_data_local["_random_"] = np.random.uniform(size = X_data_local.num_rows())
    X_data_local = X_data_local.sort("_random_")
    del X_data_local["_random_"]

    save_as_train_and_test(X_data_local, train_path % mi, valid_path % mi)

