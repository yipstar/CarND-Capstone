import matplotlib.image as mpimg
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import config
from sklearn.cross_validation import train_test_split
from collections import defaultdict


class DataSet(object):

	def __init__(self):
		"""
		log_img_paths: a list of tuple (path_to_log.csv, path_to_IMG_folder)
		"""
		self.data = defaultdict(list)
		self.data_pickle='data.p'

	def load_data(self, img_paths=config.train_data):

		for imagelist_file in img_paths:	
			with open(imagelist_file, 'r') as f:
				lines = f.readlines()

			for line in lines:
				parts = line.split(',')				
				# line =   # strip trailing newline
				# print (config.data_dir["base_dir"] + os.sep + parts[0])
				img_path = config.data_dir["base_dir"] + os.sep + parts[0]
				distance =  parts[1][:-1]
				img = mpimg.imread(img_path)
				item = [img,distance]

				# if img_path.find("red") != -1:
				# 	self.data['red'].append(zip(img,float(distance)) )
				# elif img_path.find("green") != -1:
				# 	self.data['green'].append(zip(img,float(distance)) )
				# elif img_path.find("yellow") != -1:
				# 	self.data['yellow'].append(zip(img,float(distance)) )
				# elif img_path.find("unknow") != -1:
				# 	self.data['unknow'].append(zip(img,float(distance)) )
				# else:
				# 	print("OooooooooooPs! Wrong data: ", line)

				if img_path.find("red") != -1:
					self.data['red'].append(item)
				elif img_path.find("green") != -1:
					self.data['green'].append(item)
				elif img_path.find("yellow") != -1:
					self.data['yellow'].append(item)
				elif img_path.find("unknow") != -1:
					self.data['unknow'].append(item)
				else:
					print("OooooooooooPs! Wrong data: ", line)				

	def save(self):
		# Save to pickle file
		# self.dict = {'non_vehicles': np.array(self.non_vehicles), 'vehicles': np.array(self.vehicles)}
		with open(config.data_dir["output"] + os.sep + self.data_pickle, 'wb') as f:
			pickle.dump(self.data, f)

		print("Finished save data ...")
		return self

	def restore(self, path = config.data_dir["output"] ):
		# Save to pickle file
		self.data = pickle.load(open(path + os.sep + self.data_pickle, "rb" ) )
		print("Finished restore data ...")		
		return self

	def display_hist_classes(self):
		xticks = np.arange(4)
		ind = xticks[:]
		width = 0.65

		class_counts = (len(self.data['red']), len(self.data['green']), len(self.data['yellow']),len(self.data['unknow']))
		plt.bar(ind, class_counts, width, facecolor='green', alpha=0.5, align='center')
		plt.xticks(xticks, ('red', 'green', 'yellow', 'unknow'))
		plt.yticks(np.arange(0, int(1.2*max(class_counts)), 1000))
		plt.grid(True)
		plt.title('Class Histogram')		

		# plt.savefig(config.data_dir["output_images"] + 'data_hist' + '.jpg')
		plt.savefig('data_hist.jpg')	

		plt.show()
		return		

def main():
	dataset = DataSet()
	dataset.load_data(config.train_data)
	dataset.save()	
	# dataset.restore()
	dataset.display_hist_classes()	

if __name__ == '__main__':
    main()		