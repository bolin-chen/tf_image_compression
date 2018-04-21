#!/bin/bash


while getopts b:m: option
do
  case "${option}" in
    b) base_num=${OPTARG};;
    m) model_num=${OPTARG};;
esac
done

# echo base_model/$base_num/params_test/
# echo model_$model_num/params/

cp base_model/$base_num/params_for_test/params* model_$model_num/params/
cp base_model/$base_num/params_for_test/checkpoint model_$model_num/params/
