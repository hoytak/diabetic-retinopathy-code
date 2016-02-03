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

com="convert \
    -background black \
    -fuzz 10% -trim +repage -resize $size \
    -gravity center -background black -extent $size \
    -resize $size \
    -quality 95 \
    $in_file $out_file"

echo $com

[ -e $out_file ] || $com
