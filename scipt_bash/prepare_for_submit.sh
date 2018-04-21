#!/bin/bash


while getopts m:s: option
do
  case "${option}" in
    m) model_num=${OPTARG};;
    s) submit_num=${OPTARG};;
esac
done


# echo $model_num
# echo $submit_num


mkdir submit/$submit_num/params/
cp model_$model_num/params_for_test/params* submit/$submit_num/params/
cp model_$model_num/params_for_test/checkpoint submit/$submit_num/params/
cp model_$model_num/params_for_test/model.py submit/$submit_num/
cp model_$model_num/params_for_test/config.json submit/$submit_num/


cp -r basic_block/ submit/$submit_num/
cp -r data_loader/ submit/$submit_num/
cp -r utils/ submit/$submit_num/

mkdir submit/$submit_num/data_info/
cp data_info/channel_normalization_params.npz submit/$submit_num/data_info/
cp data_info/distribution_info_$model_num.npy submit/$submit_num/data_info/
mv submit/$submit_num/data_info/distribution_info_$model_num.npy submit/$submit_num/data_info/distribution_info.npy

cp submit/decoder.py submit/$submit_num/


