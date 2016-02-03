#!/bin/bash 

# To run, this assumes that you are in the directory with the images
# unpacked into train/ and test/.  To run, it works best to use GNU
# parallel as
#
#   ls train/*.jpeg test/*.jpeg | parallel ./prep_image.sh
#
# Otherwise, it also works to do a bash for loop, but this is slower.
#
#   for f in `ls train/*.jpeg test/*.jpeg`; do ./prep_image.sh $f; done
#

size=360x360

in_file=$1
out_file=$2
random_seed=$3

get_random() {
n=`echo $1-${random_seed} | md5sum`
echo $((16#${n::12}))
}

rn=`get_random 1`
random_rotate=`expr ${rn} % 51 - 25`

rn=`get_random 2`
contrast_stretch_1=`expr ${rn} % 11 - 5`

rn=`get_random 2b`
contrast_stretch_2=`expr ${rn} % 11 - 5`

rn=`get_random 3`
random_hue=`expr 75 + ${rn} % 51`

rn=`get_random 4`
random_sat=`expr 75 + ${rn} % 51`

rn=`get_random 5`
random_val=`expr 75 + ${rn} % 51`

rn=`get_random 5b`
random_brightness=`expr ${rn} % 31 - 15`

rn=`get_random 5c`
random_contrast=`expr ${rn} % 31 - 15`

# rn=`get_random 6`
# random_sigmoidal_contrast_1=`expr 1 + ${rn} % 5`

# rn=`get_random 7`
# random_sigmoidal_contrast_2=`expr 90 + ${rn} % 10`

com="convert \
    -background black -rotate $random_rotate \
    -fuzz 10% -trim +repage -resize $size \
    -gravity center -background black -extent $size \
    -brightness-contrast ${random_brightness}x${random_contrast} \
    -contrast-stretch ${contrast_stretch_1}x${contrast_stretch_2} \
    -modulate $random_hue,$random_sat,$random_val \
    -resize $size \
    -quality 95 \
    $in_file $out_file"

echo $com

[ -e $out_file ] || $com
