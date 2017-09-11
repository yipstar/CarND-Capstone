# import process

# shift_range = 150
# steer_threshold = .15
# pr_threshold = 1

data_dir = {
	"base_dir" : "./",
	"output_images" : "../output_images/",
	"output" : "./output/" ,
	"test" : "../test/"
	}

# test_data = [
#         # data of track1 from udacity
#         # ("data/udacity/driving_log.csv", "data/udacity/IMG/")
#         ("data/didi/train_label_blank.csv", "data/didi/IMG/test")		
# ]

train_data = [
    # data of track1 from udacity
    ("./data/red.txt", "../data/red"),
    ("./data/green.txt", "../data/green"),    
    ("./data/yellow.txt", "../data/yellow"),    
    ("./data/unknow.txt", "../data/unknow")
]

## common settings
nb_epoch = 1
batch_size = 74
model_prefix = "../models/"

# xycols = ["center", "left", "right", "steer"]
xycols = ["toplidar", "tx","ty","tz"]

# comma ai setting
model_name = "comma_ai"
# image_size = (64, 64, 3)
image_size = (800, 700, 3)
loss_fn = "mse"		


# processors = {
# 				"train": process.lidar_top_processor_for_train(image_size),				
# 				"valid": process.lidar_top_processor_for_valid(image_size),
# 				"test":  process.lidar_top_processor_single_image_for_test(image_size)
# 				}

