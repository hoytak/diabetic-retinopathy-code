import graphlab as gl
import re
import random
from copy import copy
import os
import graphlab.aggregate as agg
import array

import sys

# gl.set_runtime_config("GRAPHLAB_CACHE_FILE_LOCATIONS", os.path.expanduser("~/data/tmp/"))

model_path = "/data/hoytak/diabetic/models/models/model-0-pooling-3"

train_sf = []
test_sf = []
feature_names = []
each_sf_feature_names = []

# for n in [0, "1b", '2b', 4]:
for n in [0]: #, 1, "1b", 2, '2b', 3, 4]:
    try: 
        print "Loading %s" % str(n)
        Xf_train = gl.SFrame(model_path + "/scores_train_raw")
        Xf_test = gl.SFrame(model_path + "/scores_test")

        sf_feature_names = []
        
        for fn in Xf_train.column_names():
            if fn.startswith("scores"):

                key = fn
                
                idx = 0
                while key in feature_names:
                    key = fn + ".%d" % idx
                    idx += 1

                if key != fn:
                    Xf_train.rename({fn : key})
                    Xf_test.rename({fn : key})
                
                sf_feature_names.append(key)
                
                
        train_sf.append(Xf_train)
        test_sf.append(Xf_test)
        each_sf_feature_names.append(sf_feature_names)
        feature_names += sf_feature_names
                
    except IOError, ier:
        print "Skipping %s" % str(n), ": ", str(ier)

# Train a boosted tree model on each sframe.
fn_path = "alt_test_predictions-linear/"

if False and os.path.exists(fn):

    X_train = gl.SFrame(fn_path + "/train")
    X_test = gl.SFrame(fn_path + "/test")
    
else:
    X_train = train_sf[0][["name", "level"]]
    X_test = test_sf[0][["name"]]
        
    for i, (tr_sf, te_sf, fnl) in enumerate(zip(train_sf,test_sf,each_sf_feature_names)):

        tr_2, tr_valid = tr_sf.random_split(0.97)

        print "Training model %d of %d" % (i, len(fnl))
        print fnl

        # m = gl.regression.boosted_trees_regression.create(
        #     tr_2, target = "level", features = fnl,
        #     max_iterations= 100,
        #     column_subsample=1,
        #     row_subsample=1,
        #     validation_set = tr_valid)
        
        m = gl.regression.linear_regression.create(
            tr_2, target = "level",
            features = fnl,
            max_iterations= 100, 
            validation_set = tr_valid, l2_penalty=0.02, solver='newton')
        
        # m = gl.regression.boosted_trees_regression.create(
        #     tr_2, target = "level", features = fnl,
        #     max_iterations= (400 if i == 0 else 1000),
        #     column_subsample=0.5,
        #     row_subsample=0.5,
        #     validation_set = tr_valid,
        #     step_size=0.01)
        
        pn = 'L%d' % i
        
        tr_sf[pn] = m.predict(tr_sf)
        te_sf[pn] = m.predict(te_sf)

        score_feature = [f for f in fnl if f.startswith('scores')]

        X_train = X_train.join(tr_sf[["name", pn] + score_feature], on = "name")
        X_test = X_test.join(te_sf[["name", pn] + score_feature], on = "name")

    X_train.save("alt_test_predictions/train")
    X_test.save("alt_test_predictions/test")
    
################################################################################
# Run the predictions

import numpy as np

def pred_median(d):
    return np.median([v for k, v in d.iteritems() if k.startswith('L')])

def pred_max(d):
    return max(v for k, v in d.iteritems() if k.startswith('L'))

def pred_sorted(d):
    return dict( (i, v) for i, (v, k) in enumerate(
        sorted( (v, k) for k, v in d.iteritems() if k.startswith('L'))))

X_train['median'] = X_train.apply(pred_median)
X_test['median'] = X_test.apply(pred_median)

X_train['max'] = X_train.apply(pred_max)
X_test['max'] = X_test.apply(pred_max)

X_train['sorted'] = X_train.apply(pred_sorted)
X_test['sorted'] = X_test.apply(pred_sorted)

X_train_2, X_valid = X_train.random_split(0.97)
    
prediction_method = "lr"

features = X_train.column_names()
del features[features.index("name")]
del features[features.index("level")]

if prediction_method == "lr":
            
    m = gl.regression.linear_regression.create(
        X_train_2, target = "level",
        features = features,
        max_iterations= 100, 
        validation_set = X_valid,
        solver='newton')
    
    X_test['level'] = m.predict(X_test).apply(lambda x: min(4, max(0, int(round(x)))))

elif prediction_method == "brt":
        
    m = gl.regression.boosted_trees_regression.create(
        X_train, target = "level", 
        features = features,
        max_iterations=200,
        max_depth=2,
        column_subsample=1,
        row_subsample=0.1,
        step_size=0.01)

    X_test['level'] = m.predict(X_test).apply(lambda x: min(4, max(0, int(round(x)))))
    
elif prediction_method == "median":

    X_train['level'] = X_train['median']
    X_test['level'] = X_test['median']

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
import time

with open('submission-%d.csv' % int(time.time()), 'wb') as outfile:

    fieldnames = ['image', 'level']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()
    
    for d in X_out[['image', 'level']]:
        writer.writerow(d)
    
