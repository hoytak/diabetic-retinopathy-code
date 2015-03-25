import graphlab as gl
import re
import random
from copy import copy
import os
import graphlab.aggregate as agg
import array

import sys

# gl.set_runtime_config("GRAPHLAB_CACHE_FILE_LOCATIONS", os.path.expanduser("~/data/tmp/"))

model_path = "/data/hoytak/diabetic/models/models"

train_sf = []
test_sf = []
feature_names = []

for n in [0, 1, 2, 4]:
    
    try: 
        print "Loading %s" % str(n)
        Xf_train = gl.SFrame(model_path + "/scores_train_%s" % str(n))
        Xf_test = gl.SFrame(model_path + "/scores_test_%s" % str(n))

        for fn in Xf_train.column_names():
            if fn.startswith("scores") or fn.startswith("features"):

                key = fn

                idx = 0
                while key in feature_names:
                    key = fn + ".%d" % idx
                    idx += 1

                if key != fn:
                    Xf_train.rename({fn : key})
                    Xf_test.rename({fn : key})
                
                feature_names.append(key)
                
        train_sf.append(Xf_train)
        test_sf.append(Xf_test)
                
    except IOError, ier:
        print "Skipping %s" % str(n), ": ", str(ier)

    
# Train a simple regressor to classify the different outputs 
assert train_sf

print "Joining sframes."

for sf in train_sf[1:]:
    train_sf[0] = train_sf[0].join(sf, on = ["name", "level"])
        
for sf in test_sf[1:]:
    test_sf[0] = test_sf[0].join(sf, on = "name")

print "Generating random split."
X_train, X_valid = train_sf[0].random_split(0.97)
X_test = test_sf[0]

m = gl.regression.boosted_trees_regression.create(
    X_train, target = "level", features = feature_names,
    max_iterations=200, 
    validation_set=X_valid,
    column_subsample=0.2,
    row_subsample=0.5,
    step_size=0.025)

# m = gl.regression.linear.create(
#     X_train, target = "level", features = feature_names,
#     # max_iterations=200, 
#     validation_set=X_valid,
#     column_subsample=0.2,
#     row_subsample=1,
#     step_size=0.025)
 

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
    
