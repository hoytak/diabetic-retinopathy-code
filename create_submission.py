import graphlab as gl
import re
import random
from copy import copy
import os
import graphlab.aggregate as agg
import array

import sys

# gl.set_runtime_config("GRAPHLAB_CACHE_FILE_LOCATIONS", os.path.expanduser("~/data/tmp/"))

base_path = os.getcwd()

model_path = base_path + "/nn_256x256/models/"

train_sf = []
test_sf = []
feature_names = []

for n in [0,1,2,3,4]:
    
    try: 
        Xf_train = gl.SFrame(model_path + "/scores_train_%d" % n)
        Xf_test = gl.SFrame(model_path + "/scores_test_%d" % n)

        train_sf.append(Xf_train)
        test_sf.append(Xf_test)
        
        feature_names += ["scores_%d" % n, "features_%d" %n]
        
    except IOError, ier:
        print "Skipping %d" % n, ": ", str(ier)

    
# Train a simple regressor to classify the different outputs 
assert train_sf

for sf in train_sf[1:]:
    train_sf[0] = train_sf[0].join(sf, on = ["name", "level"])
        
for sf in test_sf[1:]:
    test_sf[0] = test_sf[0].join(sf, on = "name")

X_train, X_valid = train_sf[0].random_split(0.95)
X_test = test_sf[0]

m = gl.regression.boosted_trees_regression.create(
    X_train, target = "level", features = feature_names,
    max_iterations=500, validation_set=X_valid,
    column_subsample=0.2, row_subsample=1, step_size=0.01)

X_test['level'] = m.predict(X_test).apply(lambda x: min(4, max(0, int(round(x)))))

X_out = X_test[['name', 'level']]

def get_number(s):
    n = float(re.match('[0-9]+', s).group(0))
    if 'right' in s:
        n += 0.5
    return n
    
X_out['number'] = X_out['name'].apply(get_number)
X_out = X_out.sort('number')
X_out.rename({"name" : "image"})

import csv

with open('submission.csv', 'wb') as outfile:

    fieldnames = ['image', 'level']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()
    
    for d in X_out[['image', 'level']]:
        writer.writerow(d)
    
