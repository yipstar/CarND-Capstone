# unzip the zip file provided by binliu
# move to the directory just traffic light
# make two directories there -> site_training and site_validation

import os
import shutil

train = pd.read_csv('./site_train.csv')
valid = pd.read_csv('./site_valid.csv')

for i in range(len(train)):
    shutil.copy('./camera'+ train.ix[i]['image'], './site_training/')

for i in range(len(valid)):
    shutil.copy('./camera'+ valid.ix[i]['image'], './site_validation/')
