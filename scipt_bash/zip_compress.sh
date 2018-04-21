#!/bin/bash

# zip -r submit/$1/decoder.zip submit/$1/** -x submit/$1/encoded/**\*
zip -r submit/$1/decoder.zip submit/$1/** -x submit/$1/test_encoded/**\*

