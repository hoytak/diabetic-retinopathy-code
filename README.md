# diabetic-retinopathy-code

Code for the Kaggle competition http://www.kaggle.com/c/diabetic-retinopathy-detection.

This code uses the ImageMagick convert tool to preprocess the images,
then uses the neural net toolkits and boosted tree regression toolkits in Dato's
Graphlab Create package to build the classifier. 

# Main Ideas

1. Use ImageMagick's convert tool to trim off the blank space to the
   sides of the images, then pad them so that they are all 256x256.  Thus
   the eye is always centered with edges against the edges of the
   image. I avoided scaling the images to improve the neural net performance.

2. Create multiple versions of each image varying by hue and contrast
   and white balance.

3. Duplicate each class so each class is represented equally, then
   shuffle the data.

4. Train several neural nets, one trained to predict the level
   membership, and another 4 to distinguish 0 vs 1-4, 0-1 vs. 2-4, 0-2
   vs. 3-4, and 0-3 vs. 4.

5. For each image, extract both class predictions and the values of the
   final neural net layer.  Pool all of these features across models and
   variations of the same images.

6. Train boosted regression trees on these pooled features to predict the level. 

7. Round to the nearest integer as the class prediction. 

Each of these steps could definitely be tuned for greater accuracy,
and I am welcome to feedback.


# Quick start guide

All of the scripts below are fairly rough, and I can't guarantee they
will work right away.  I recommend reading through them and editing
them as appropriate.  I try to specify paths at the top of the files,
and these will likely need to be edited.

Getting to a solution quickly: Currently, the processing is pretty
intensive, and everything below will take several days to run.  To get
to a decent solution more quickly, comment out all but the first image
transformation in prep_image.sh.  The resulting model will not be as
good, but will take much less time to train.  


1. Install the ImageMagick convert command line tool (named
   imagemagick) and the GNU parallel tool (named parallel).  These are
   in all the major linux repositories.

2. Install the GPU version of Graphlab Create with:

   ```
   sudo pip install http://static.dato.com/files/graphlab-create-1.3.gpu.tar.gz
   ```

   See https://dato.com/products/create/gpu_install.html and
   https://dato.com/products/create/quick-start-guide.html for more info.

3. Download the data and unpack it into a folder (say diabetic/) with
   train/, test/, and trainLabels.csv.

2. Run prep_image.sh on each image to prepare the image variations and
   resized images.  The instructions are at the top of the
   prep_image.sh file. Essentially, in the directory diabetic/, run::

   ```
     ls train/*.jpeg test/*.jpeg | parallel ./prep_image.sh
   ```

   This will preprocess all the images into diabetic/processed/train/
   and diabetic/processed/test/.

3. Run create_image_sframes.py in diabetic/ to build the primary image
   SFrames.  This will produce directories
   diabetic/image-sframes/train/ and diabetic/image-sframes/test/.

4. Run create_balanced_classes.py in diabetic/ to build a collection
   of balanced classes for training each of the neural nets above.
   This may take a while, and it produces roughly 150GB of data in the
   end.  If an error occurs, the script can be restarted. 
 
5. Set the which_model variable at the top of create_nn_model.py; this
   is one of [0,1,2,3,4].  Run this file on a machine with cuda installed
   to run the NN training code.  I get about 200 images / second on
   amazon's GPU instance type, and about 320 images / second on a GTX 780.

   It takes about a day to train one of the models, but once trained,
   it can be loaded and used for any future predictions.  Models are
   saved to diabetic/nn_256x256/models/.

6. Once the model results are saved, run create_submission.py to build
   the final regression model and submission.
   
