#!/bin/bash

cp model_$1/params/* model_$1/params_for_test/
cp model_$1/model.py model_$1/params_for_test/
cp model_$1/config.json model_$1/params_for_test/
