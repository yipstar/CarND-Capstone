import matplotlib.image as mpimg
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
		# self.non_vehicles = []
		# self.vehicles = []

		# self.red = []
		# self.green = []		
		# self.yellow = []				
		# self.unknow = []						
		# # self.dict = {}
		# self.dict =  defaultdict(lambda: defaultdict(list))
		# self.data = defaultdict(lambda: defaultdict(list))
		self.data = defaultdict(list)
		 
		self.data_pickle='data.p'

	def load_data(self, img_paths):
		for imagelist_file, image_folder in img_paths:

			with open(imagelist_file, 'r') as f:
				lines = f.readlines()

			for line in lines:
				line = line[:-1]  # strip trailing newline

				# img = cv2.imread(line)
				# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				# print(line) 								
				img = mpimg.imread(config.data_dir["base_dir"] + line)


				if line.find("red") != -1:
					self.data['red'].append(img)
				elif line.find("green") != -1:
					self.data['green'].append(img)
				elif line.find("yellow") != -1:
					self.data['yellow'].append(img)
				elif line.find("unknow") != -1:
					self.data['unknow'].append(img)
				else:
					print("OooooooooooPs! Wrong data: ", line)
				

	def save(self):
		# Save to pickle file
		# self.dict = {'non_vehicles': np.array(self.non_vehicles), 'vehicles': np.array(self.vehicles)}
		with open(config.data_dir["output"] + self.data_pickle, 'wb') as f:
			pickle.dump(self.data, f)

		print("Finished save data ...")
		return self

	def restore(self):
		# Save to pickle file
		self.data = pickle.load(open(config.data_dir["output"] + self.data_pickle, "rb" ) )
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