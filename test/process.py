"""This implements several image processing methods for different models.
The common interface to the methods:
- `image_size` parameter specifies the target shape of outputs
- each processor takes an `img_io` (a image file path or BytesIO str) as input
and generates the processed image as numpy.array.
"""


import cv2
import math
import numpy as np
import random
import matplotlib.pyplot as plt

# from keras.preprocessing.image import load_img, img_to_array
# from keras.applications import imagenet_utils

import config


def lidar_top_processor_for_train(image_size):

	h, w, nch = image_size
	def fn(line_data):
		# lidar_top_filename,tx,ty,tz
		path_file = line_data['lidar_top_filename'].strip()
		image = cv2.imread(path_file)	    
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
		# image = _crop_and_resize_image(image)   
		image_arr = np.array(image)     
		tx_in = float(line_data['tx'].strip())
		ty_in = float(line_data['ty'].strip())		
		tz_in = float(line_data['tz'].strip())				
		tx_in= float(tx_in)		
		ty_in= float(ty_in)		
		tz_in= float(tz_in)						
		return image_arr, tx_in, ty_in, tz_in 

	return fn


def lidar_top_processor_for_valid(image_size):

	h, w, nch = image_size
	def fn(line_data):
		path_file = line_data['lidar_top_filename'].strip()
		image = cv2.imread(path_file)	    
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
		# image = _crop_and_resize_image(image)   
		image_arr = np.array(image)     
		tx_in = float(line_data['tx'].strip())
		ty_in = float(line_data['ty'].strip())		
		tz_in = float(line_data['tz'].strip())				
		tx_in= float(tx_in)		
		ty_in= float(ty_in)		
		tz_in= float(tz_in)						
		return image_arr, tx_in, ty_in, tz_in 

	return fn

def lidar_top_processor_single_image_for_test(image_size):

	h, w, nch = image_size
	def fn(image_path):
		path_file = image_path.strip()
		image = cv2.imread(path_file)	    
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
		image = _crop_and_resize_image(image)   
		image_arr = np.array(image)     
		return image_arr 

	return fn


# commands = {'shift': 'shift', 'resize': 'resize', 'flip': 'flip','none': 'none','brightness' : 'brightness'}

def process_image_for_report(image,command_q):	

	# image = cv2.imread(image_path)	    
	# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 

	# steer = float(steer)	

	while command_q.qsize() > 0:
		command = command_q.get()

		if  command == commands['none']:
			command_q.empty()    

		if  command == commands['shift']:
			image,steer,tr_x = _shift_image(image,steer,lane_id)

		if  command == commands['resize']:
			image = _crop_and_resize_image(image)

		if  command == commands['flip']:
			image,steer = _flip_image(image,steer)

		if  command == commands['brightness']:
			image = _brightness_image(image)

	return image,steer


# def _shift_image(image,steer,lane_id):    
#     # Translation
# 	steer = float(steer)
# 	rows,cols = image.shape[:2]
# 	trans_range = config.shift_range
# 	tr_x = trans_range*np.random.uniform()-trans_range/2
# 	steer_ang = steer + tr_x/trans_range*2*.2
# 	# steer_ang = steer + tr_x/trans_range*2*.3	
# 	tr_y = 10*np.random.uniform()-10/2

# 	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
# 	image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))

# 	return image_tr,steer_ang,tr_x

def _brightness_image(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def _crop_and_resize_image(image):
    # Preprocessing image files
    col,row,ch = config.image_size
    cropped_image = image[55:135, :, :]    
    shape = cropped_image.shape
    # note: numpy arrays are (row, col)!
    image = cropped_image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(col,row), interpolation=cv2.INTER_AREA)    
    return image     

def _flip_image(image,steer):
	steer = float(steer)
	prob_f = np.random.random()
	if prob_f > 0.5:
		# # flip the image and reverse the steering angle
		steer_out = -1*steer
		image_out =  cv2.flip(image, 1)
	else:
		image_out, steer_out = image, steer
	return image_out, steer_out

# def plot_images(images,steers,titles,deg = 0):
#     plt.subplot(1,3,1)
#     plt.imshow(images[0]);
#     plt.axis('on')    
#     if deg==0:
#         plt.title(titles[0] + ' ' +'steer:'+ str(np.round(steers[0],2) ) );        
#     else:
#         plt.title('Steer:'+ str((np.round((steer+.1)*180/np.pi,2) )))
#     plt.subplot(1,3,2)
#     plt.imshow(images[1]);
#     plt.axis('on')        
#     if deg==0:
#         plt.title('Steer:'+ str(np.round(steers[1],2) ));
#         plt.title(titles[1] + ' ' +'steer:'+ str(np.round(steers[1],2) ) );                
#     else:
#         plt.title('Steer:'+ str(np.round(steer*180/np.pi,2) ));
#     plt.subplot(1,3,3)
#     plt.imshow(images[2]);    
#     plt.axis('on')        
#     if deg==0:
#         plt.title(titles[2] + ' ' +'steer:'+ str(np.round(steers[2],2) ) );        
#     else:
#         plt.title('Steer:'+ str((np.round((steer-.1)*180/np.pi,2) )))     

def plot_images(images,labes):
		plt.subplot(1,len(images),1)
		count = 1
		for image,label in zip(images,labes):
			plt.subplot(1,len(images),count)			
			plt.imshow(image)
			plt.title(label)			
			plt.axis('on')
			count +=1
