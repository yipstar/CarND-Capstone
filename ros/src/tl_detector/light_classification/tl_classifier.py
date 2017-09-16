from styx_msgs.msg import TrafficLight
import rospy
import os
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import scipy
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):

        model_file = "./light_classification/models/model.json"
        weights_file = "./light_classification/models/model.h5"

        # model_file = "./light_classification/models/model_squeezenet.json"
        # weights_file = "./light_classification/models/model_squeezenet2.h5"

        #load json and create model
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(weights_file)
        rospy.logwarn("Loaded model from disk")

        self.model = loaded_model

        # See https://github.com/fchollet/keras/issues/2397
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        # See https://github.com/fchollet/keras/issues/2397
        with self.graph.as_default():

            # evaluate loaded model on test data
            model = self.model

            new_shape = (224, 224, 3)

            image_resized = scipy.misc.imresize(image, new_shape)
            img = image_resized
            # img = image_resized.transpose((-1, 0, 1))

            x = np.expand_dims(img, axis=0)
            # rospy.logwarn("shape of x: %s", x.shape)

            predictions = model.predict(x)
            predicted_label = np.argmax(predictions)

            rospy.logwarn("predicted_label: %s", predicted_label)

            # Labels need to be converted the labeled dataset labels don't match the TrafficLight refs.
            # In TrafficLgiht UNKNOWN=4, GREEN=2, YELLOW=1, and RED=0

            if predicted_label == 1:
                predicted_label = 2
            elif predicted_label == 2:
                predicted_label = 1

            return predicted_label
            # return TrafficLight.UNKNOWN
