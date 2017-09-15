# From PROJECT_ROOT/data, download: https://s3-us-west-2.amazonaws.com/traffic-light-dataset/traffic_light_170906.zip and unzip

# Then run this script from its current directory

import pandas as pd
import os
import shutil

root_data_dir = "../../../../../../data/traffic_light_170906/data/"

data_dir = os.path.abspath(root_data_dir)
print(data_dir)

train_df = pd.read_csv('train.csv')
train_dir = "./training/"

valid_df = pd.read_csv('valid.csv')
valid_dir = "./validation/"

folder_map = {
    0: data_dir + '/red/',
    1: data_dir + '/green/',
    2: data_dir + '/yellow/',
    3: data_dir + '/unknow/'
}

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

for index, row in train_df.iterrows():
    print(row)
    filename = row[0]
    label = row[1]
    src = ""
    src = folder_map[label] + filename
    dest = train_dir + filename
    shutil.copy(src, dest)

if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

for index, row in valid_df.iterrows():
    print(row)
    filename = row[0]
    label = row[1]
    src = ""
    src = folder_map[label] + filename
    dest = valid_dir + filename
    shutil.copy(src, dest)
