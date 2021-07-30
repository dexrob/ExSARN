#!/usr/bin/env bash

echo "Download compiled data from git repo!"
# URL=https://github.com/dexrob/BioTac_slide_20_50/tree/master/compiled_data
URL=https://github.com/dexrob/BioTac_slide_20_50

DIR=/Users/ruihan/Documents/IROS2020/Supervised-Autoencoder-Joint-Learning-on-Heterogeneous-Tactile-Sensory-Data/data
mkdir $DIR

cd $DIR

git init

git remote add origin -f $URL

git config core.sparseCheckout true # enable this

echo "begin copy folder"

echo compiled_data/*> .git/info/sparse-checkout # if you don’t start with root you are liable to download multiple folders wherever the name you specified is matched.

echo "end copy folder, begin pull"
git pull origin master

# now, the compiled_data folder should appears at $DIR/compiled_data

# to view the raw data and more information about the materials,
# git clone $URL

