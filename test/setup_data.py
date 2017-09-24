import sys,os
import os.path
import glob
import re
import pandas as pd
import argparse
import shutil
sys.path += [os.path.join(os.path.dirname(__file__), '..') +'/ros/src/tl_detector']
# from traffic_light_config import config
from collections import defaultdict
from fnmatch import fnmatch
from sklearn.utils import shuffle

def main():
    parser = argparse.ArgumentParser(description='prepare training data.')

    parser.add_argument('-i', '--indir', type=str, nargs='?', default='data',
        help='Input folder where raw are located')        

    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='output',
        help='Output folder')    

    parser.add_argument('-csv', '--csv', type=str, nargs='?', default='data.csv',
        help='Input folder where output are located')        

    args = parser.parse_args()
    csvfile = args.csv
    outdir = args.outdir
    indir = args.indir

    dataset_name = 'traffic_light_large_170918'
    cols = ['filename','distance']
    # path = indir + os.sep + dataset_name + os.sep + csvfile
    path = indir + os.sep + dataset_name 
    csvfile = path + os.sep + csvfile

    # path = os.path.abspath(os.path.join(os.getcwd(), indir + os.sep + dataset_name + os.sep + csvfile) )    

    df = pd.read_csv(csvfile, names = cols,header =None)
    pd.set_option('display.width', 120)   
    # print df['filename']  

    data = defaultdict(pd.DataFrame)
    # data['red'] = df[ df['filename'].find('red') != -1  ]
    # data['green'] = df[ df['filename'].find('green') != -1  ]
    # data['yellow'] = df[ df['filename'].find('yellow') != -1  ]
    # data['unknow'] = df[ df['filename'].find('unknow') != -1  ]

    # rlst = [x for x in df['filename'] if x.find('red') != -1]
    # glst = [x for x in df['filename'] if x.find('green') != -1]    
    # ylst = [x for x in df['filename'] if x.find('yellow') != -1]    
    # ulst = [x for x in df['filename'] if x.find('unknow') != -1]        

    rlst = [ (x.split('/')[-1], 0) for x in df['filename'] if x.find('red') != -1 ]    
    glst = [ (x.split('/')[-1], 2) for x in df['filename'] if x.find('green') != -1 ]        
    ylst = [ (x.split('/')[-1], 1) for x in df['filename'] if x.find('yellow') != -1 ]
    ulst = [ (x.split('/')[-1], 4) for x in df['filename'] if x.find('unknow') != -1 ]    

    data['red'] = pd.DataFrame(rlst, columns=['image','label'])
    data['green'] = pd.DataFrame(glst, columns=['image','label'])
    data['yellow'] = pd.DataFrame(ylst, columns=['image','label'])
    # data['unknow'] = pd.DataFrame(ulst, columns=['image','label'])

    # full_df = pd.concat([data['red'],data['green'],data['yellow'],data['unknow']])    
    full_df = pd.concat([data['red'],data['green'],data['yellow']])        
    full_df = shuffle(full_df)

    train_dir = "training"
    valid_dir = "validation"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    # train_df = full_df.iloc[:full_df.shape[0]*frac]
    # valid_df = full_df.iloc[full_df.shape[0]*frac:]
    train_df = full_df.iloc[:3000]
    valid_df = full_df.iloc[3001:]


    # print train_df.shape[0]
    # print valid_df.shape[0]

    # print train_df.head
    # print valid_df.head

    # In TrafficLgiht UNKNOWN=4, GREEN=2, YELLOW=1, and RED=0
    folder_map = {
        0: path + '/red/',
        2: path + '/green/',
        1: path + '/yellow/',
        4: path + '/unknow/'
    }    

    train_df.to_csv(outdir + os.sep + 'train.csv', index=False)   

    for index, row in train_df.iterrows():
        filename = row[0]
        label = row[1]
        src = ""
        src = folder_map[label] + filename
        print(src)

        if os.path.isfile(src):
            dest = outdir + os.sep + train_dir + os.sep + filename
            shutil.copy(src, dest)


    valid_df.to_csv(outdir + os.sep + 'valid.csv', index=False)   
    for index, row in valid_df.iterrows():
        filename = row[0]
        label = row[1]
        src = ""
        src = folder_map[label] + filename

        if os.path.isfile(src):
            dest = outdir + os.sep + valid_dir + os.sep + filename            
            shutil.copy(src, dest)


if __name__ == '__main__':
    main()