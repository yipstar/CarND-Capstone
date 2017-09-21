import sys,os
import os.path
import glob
import config 
import re
import pandas as pd
import argparse
import shutil
sys.path += [os.path.join(os.path.dirname(__file__), '..') +'/ros/src/tl_detector']
from traffic_light_config import config
from collections import defaultdict
from fnmatch import fnmatch


def clean_csv(indir,outdir,df,csvfile):

    path = os.path.abspath(os.path.join(os.getcwd(), indir,'delete'))     
    # del_lst =  [(x.split('.')[-2], x)  for x in os.listdir(path) if fnmatch(x, '*.jpg') ] 
    del_lst =  [x.split('.')[-2] for x in os.listdir(path) if fnmatch(x, '*.jpg') ]     

    print("Before clean, the dataframe size is {}".format(df.shape))
    df.drop(del_lst,inplace=True)        
    print("After clean, the dataframe size is {}".format(df.shape))
    path = os.path.abspath(os.path.join(os.getcwd(), indir))         
    df.to_csv(path + os.sep + csvfile+'.new')    

    print('successful clean csv files with {} rows!'.format(len(del_lst)))                 

    path_del = os.path.join(indir, 'delete')
    del_lst = ['camera/'+x for x in os.listdir(path_del) if fnmatch(x, '*.jpg') ]
    del_df = pd.DataFrame(del_lst,columns=['filename'])

    move_images(del_df,indir,path_del)
    print("successful clean {} images!".format(del_df.shape)) 

def  split(indir,outdir,data):

    color = ['red','green','yellow','unknow']
    dest = [outdir + os.sep + x for x in color ]

    for folder in dest: 
        if not os.path.exists(folder):
            os.makedirs(folder)    
    
    for fname in data['red']['filename']:
        source = indir + os.sep + fname
        shutil.copy(source,dest[0])    

    for fname in data['green']['filename']:
        source = indir + os.sep + fname
        shutil.copy(source,dest[1])    

    for fname in data['yellow']['filename']:
        source = indir + os.sep + fname
        shutil.copy(source,dest[2])  

    for fname in data['unknow']['filename']:
        source = indir + os.sep + fname
        shutil.copy(source,dest[3])  

    data['red']['filename'] = [x.replace('camera','red') if 'camera' in x else x for x in data['red']['filename']]    
    data['green']['filename'] = [x.replace('camera','green') if 'camera' in x else x for x in data['green']['filename']]    
    data['yellow']['filename'] = [x.replace('camera','yellow') if 'camera' in x else x for x in data['yellow']['filename']]    
    data['unknow']['filename'] = [x.replace('camera','unknow') if 'camera' in x else x for x in data['unknow']['filename']]                

    for x in color: 
        # data[x][['filename','distance']].to_csv(outdir+'data.txt',index=False)    
        data[x][['filename','distance']].to_csv(outdir + os.sep +  'data.csv' , mode='a', header=False,index=False)            

    print('successful split data!') 

def merge(indir,outdir):

    img_dest = os.path.join(outdir, 'all/camera')
    csv_dest = os.path.join(outdir, 'all/cap_camera.csv')
    cols = ['timestamp','filename','light','state','distance']    

    if not os.path.exists(img_dest):
        os.makedirs(img_dest)       
   
    folders = [name for name in os.listdir(indir) if os.path.isdir(os.path.join(indir, name ))]
    
    for name in folders[1:]:
        img_source = os.path.join(indir, name + os.sep +'camera')
        print ('img_source = {}, img_dest = {}'.format(img_source,img_dest))
        imgfiles = [x  for x in os.listdir(img_source) if fnmatch(x, '*.jpg') ] 
        for img in imgfiles:            
            shutil.copy(img_source + os.sep + img, img_dest)  
    
        csv_source = os.path.join(indir, name + os.sep +'cap_camera.csv')
        df = pd.read_csv(csv_source, names = cols,header =None)        
        df.to_csv(csv_dest , mode='a', header=None,index=False)                        

    print('successful merge data!') 

def clean_images(indir,outdir,df,csvfile):
    
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    large_dist_df = df[ (df['distance'] > 60.) & (df['distance'] < 1000 ) ]    
    neg_dist_df  =   df[ (df['distance'] < 0.) ]
    del_df = pd.concat([large_dist_df,neg_dist_df])
    
    img_deleted = []
    path = os.path.join(indir, 'all')
    jpgfiles = [x  for x in del_df['filename'] if fnmatch(x, '*.jpg') ]     

    path_del = os.path.join(indir, 'delete')
    move_images(del_df,indir,path_del)
    print("successful clean {} images!".format(del_df.shape)) 
  
def move_images(del_df,indir,path_del): 
    if not os.path.exists(path_del):
        os.makedirs(path_del)       
    
    for name in del_df['filename']:
        source = indir + os.sep + name
        parts = re.split('(\\\\|/)', name)
        filename = parts[-1]

        if(os.path.isfile(source)):
            if (os.path.isfile(path_del + os.sep + filename)):
                os.remove(source)
            else:
                shutil.move(source,path_del)  

    print("successful move {} images to delete dir!".format(del_df.shape))         

def main():
    parser = argparse.ArgumentParser(description='wrangle the raw data that extract from bag.')

    parser.add_argument('-i', '--indir', type=str, nargs='?', default='output\ship\\all',
        help='Input folder where raw are located')        

    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='output\ship\processed-all',
        help='Output folder')    
    parser.add_argument('-csv', '--csv', type=str, nargs='?', default='cap_camera.csv',
        help='Input folder where output are located')

    group = parser.add_argument_group('data process')
    group.add_argument('-op', '--operation', type=str, nargs='?', default='clean_csv', help='merge,split,clean_images,clean_csv')

    args = parser.parse_args()
    csvfile = args.csv
    operation = args.operation
    outdir = args.outdir
    indir = args.indir

    if operation == 'merge':
       merge(indir,outdir)
       exit(0)

    cols = ['timestamp','filename','light','state','distance']
    df = pd.read_csv(indir + os.sep + csvfile, names = cols,header =None, index_col='timestamp')
    pd.set_option('display.width', 120)    

    data = defaultdict(pd.DataFrame)
    data['red'] = df[ df['state'] == str(config.light_state['RED']) ]
    data['green'] = df[ df['state'] ==  str(config.light_state['GREEN']) ]
    data['yellow'] = df[ df['state'] ==  str(config.light_state['YELLOW']) ] 
    data['unknow'] =  df[ df['state'] == str(config.light_state['UNKNOW']) ]

    if operation == 'split':
       split(indir,outdir,data)
    elif operation == 'clean_csv':
       clean_csv(indir,outdir,df,csvfile)
    elif operation == 'clean_images':
       clean_images(indir,outdir,df,csvfile)

if __name__ == '__main__':
    main()
