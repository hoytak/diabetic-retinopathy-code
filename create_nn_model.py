import graphlab as gl
import re
import random
from copy import copy
import os
import graphlab.aggregate as agg
import array

import sys

model_name = "pooling-2"
which_model = 0

print "Running model %d, %s" % (which_model, model_name)

alt_path = os.path.expanduser("~/data/tmp/")
if os.path.exists(alt_path):
    gl.set_runtime_config("GRAPHLAB_CACHE_FILE_LOCATIONS", alt_path)

model_path = "nn_256x256/models/model-%d-%s/" % (which_model, model_name)
model_filename = model_path + "nn_model" 

X_train = gl.SFrame("image-sframes/train-%d/" % which_model)
X_valid = gl.SFrame("image-sframes/validation-%d/" % which_model)
X_test = gl.SFrame("image-sframes/test/")

################################################################################

# init_random vs random_type in ConvolutionLayer. 

dll = gl.deeplearning.layers

nn = gl.deeplearning.NeuralNet()

nn.layers.append(dll.ConvolutionLayer(
    kernel_size = 5,
    stride=2,
    num_channels=96,
    init_random="xavier"))

nn.layers.append(dll.MaxPoolingLayer(
    kernel_size = 3,
    stride=2))

nn.layers.append(dll.SigmoidLayer())

for i in xrange(5):

    nn.layers.append(dll.ConvolutionLayer(
        kernel_size = 3,
        padding = 1,
        stride=1,
        num_channels=64 - 8 * i,
        init_random="xavier"))
    
    nn.layers.append(dll.SumPoolingLayer(
        kernel_size = 3,
        padding = 1,
        stride=2))

    nn.layers.append(dll.RectifiedLinearLayer())
    
# nn.layers.append(dll.ConvolutionLayer(
#     kernel_size = 8,
#     stride=4,
#     num_channels=32,
#     init_random="gaussian"))

# nn.layers.append(dll.RectifiedLinearLayer())

# nn.layers.append(dll.SigmoidLayer())

nn.layers.append(dll.FlattenLayer())

nn.layers.append(dll.FullConnectionLayer(
    num_hidden_units = 64,
    init_sigma = 0.005,
    init_bias = 1,
    init_random = "gaussian"))

nn.layers.append(dll.RectifiedLinearLayer())

nn.layers.append(dll.FullConnectionLayer(
    num_hidden_units = 32,
    init_sigma = 0.005,
    init_bias = 1,
    init_random = "gaussian"))

nn.layers.append(dll.RectifiedLinearLayer())

nn.layers.append(dll.FullConnectionLayer(
    num_hidden_units = 5 if which_model == 0 else 2,
    init_sigma = 0.005,
    init_random = "gaussian"))

nn.layers.append(dll.SoftmaxLayer())

nn.params["batch_size"] = 32
nn.params["momentum"] = 0.9
# nn.params["learning_rate"] = 0.01
# nn.params["l2_regularization"] = 0.0005
# nn.params["bias_learning_rate"] = 0.02
# nn.params["learning_rate_schedule"] = "exponential_decay"
# nn.params["learning_rate_gamma"] = 0.1
# nn.params["init_random"] = "gaussian"
# nn.params["init_sigma"] = 0.01

################################################################################

if os.path.exists("image-sframes/mean_image"):
    mean_image_sf = gl.SFrame("image-sframes/mean_image")
    mean_image = mean_image_sf["image"][0]
else:
    mean_image = X_train["image"].mean()
    mean_image_sf = gl.SFrame({"image" : [mean_image]})
    mean_image_sf.save("image-sframes/mean_image")

if which_model == 0:

    m = gl.classifier.neuralnet_classifier.create(
        X_train, features = ["image"], target = "level",
        network = nn, mean_image = mean_image,
        device = "gpu", random_mirror=True, max_iterations = 10,
        validation_set=X_valid,
        model_checkpoint_interval = 1,
        model_checkpoint_path = model_filename + "-checkpoint")
    
else:
    assert which_model in [1,2,3,4]

    X_train["class"] = (X_train["level"] >= which_model)
    X_valid["class"] = (X_valid["level"] >= which_model)

    # Downsample the less common class
    n_class_0 = (X_train["class"] == 0).sum()
    n_class_1 = (X_train["class"] == 1).sum()
    
    m = gl.classifier.neuralnet_classifier.create(
        X_train,
        features = ["image"], target = "class",
                                                  # network = nn,
        mean_image = mean_image,
        model_checkpoint_path = model_filename + "-checkpoint",
        model_checkpoint_interval = 1,
        device = "gpu", random_mirror=True, max_iterations = 10, validation_set=X_valid)
    
m.save(model_filename)

X_train["class_scores"] = \
  (m.predict_topk(X_train[["image"]], k= (5 if which_model == 0 else 2))\
   .unstack(["class", "score"], "scores").sort("row_id")["scores"])

X_test["class_scores"] = \
    (m.predict_topk(X_test[["image"]], k=(5 if which_model == 0 else 2))
     .unstack(["class", "score"], "scores").sort("row_id")["scores"])
    
X_train["features"] = m.extract_features(X_train[["image"]])
X_test["features"] = m.extract_features(X_test[["image"]])

def flatten_dict(d):
    out_d = {}

    def _add_to_dict(base, out_d, d):

        if type(d) in [array.array, list]:
            for j, v in enumerate(d):
                new_key = str(j) if base is None else (base + ".%d" % j)
                _add_to_dict(new_key, out_d, v)

        elif type(d) is dict:
            for k, v in d.iteritems():
                new_key = k if base is None else (base + '.' + str(k))
                _add_to_dict(new_key, out_d, v)
        else:
            if d != 0:
                out_d[base] = d
                    
    _add_to_dict(None, out_d, d)
    
    return out_d

score_column = "scores_%d" % which_model
features_column = "features_%d" % which_model
    
Xt = X_train[["name", "source", "class_scores", "level", "features"]]
Xty = Xt.groupby(["name", "level"], {"cs" : agg.CONCAT("source", "class_scores")})
Xty[score_column] = Xty["cs"].apply(flatten_dict)

Xty2 = Xt.groupby("name", {"ft" : agg.CONCAT("source", "features")})
Xty2[features_column] = Xty2["ft"].apply(flatten_dict)

Xty = Xty.join(Xty2[["name", features_column]], on = "name")

Xty[["name", score_column, "level", features_column]].save(model_path + "scores_train")

Xtst = X_test[["name", "source", "class_scores", "features"]]
Xtsty = Xtst.groupby("name", {"cs" : agg.CONCAT("source", "class_scores")})
Xtsty[score_column] = Xtsty["cs"].apply(flatten_dict)

Xtsty2 = Xtst.groupby("name", {"ft" : agg.CONCAT("source", "features")})
Xtsty2[features_column] = Xtsty2["ft"].apply(flatten_dict)

Xtsty = Xtsty.join(Xtsty2[["name", features_column]], on = "name")

Xtsty[["name", score_column, features_column]].save(model_path + "scores_test")



