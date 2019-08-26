#!/usr/bin/env bash

python bbox_setup.py install

echo "build solvers..."
python bbox_setup.py build_ext --inplace

echo "build psroi_pooling..."
cd models/psroi_pooling
sh make.sh
cd ../..

cd models/correlation
sh make.sh
cd ../..

cd models/roi_align
sh make.sh
cd ../..