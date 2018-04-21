#!/bin/bash


while getopts m:g: option
do
  case "${option}" in
    m) model_num=${OPTARG};;
    g) gpu_num=${OPTARG};;
esac
done


# echo $gpu_num
# echo $model_num


./encode.py -m $model_num -g $gpu_num

echo Encode complete -----

./decode.py -m $model_num -g $gpu_num

echo Decode complete -----

./processing_utils/evaluate.py -m $model_num

