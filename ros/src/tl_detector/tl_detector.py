#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import numpy as np
import math
import yaml

STATE_COUNT_THRESHOLD = 1
DISTANCE_TO_TRAFFIC_LIGHT_TO_START_CLASSIFYING = 50.0

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        final_waypoints_sub = rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.config_lights = []
        for point in self.config['stop_line_positions']:
            light = self.create_light(point[0], point[1])
            self.config_lights.append(light)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def final_waypoints_cb(self, waypoints):
        self.final_waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # rospy.logwarn("light_wp: %s, state: %s", light_wp, state)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        # self.camera_image.encoding = "rgb8"
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image
        # use x y to crop around location

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        config_lights = self.config_lights
        topic_lights = self.lights

        if(self.pose and self.waypoints and config_lights and self.lights):

            pose = self.pose
            waypoints = self.waypoints.waypoints

            closest_wp_index = self.get_closest_waypoint(pose.pose, waypoints)
            # closest_wp_index = self.get_next_point_index(pose.pose, waypoints)
            closest_wp = waypoints[closest_wp_index]

            # rospy.logwarn("closest_wp index:%s, x:%s, y:%s, car_x: %s, car_y: %s", closest_wp_index, closest_wp.pose.pose.position.x, closest_wp.pose.pose.position.y, pose.pose.position.x, pose.pose.position.y)

            # #TODO find the closest visible traffic light (if one exists)
            index = self.get_next_point_index(pose.pose, topic_lights)
            topic_light = topic_lights[index]
            next_light_wp_index_from_topic = self.get_next_point_index(topic_light.pose.pose, waypoints)

            index = self.get_next_point_index(pose.pose, config_lights)
            config_light = config_lights[index]
            next_light_wp_index_from_config = self.get_next_point_index(config_light.pose.pose, waypoints)

            # if next_light_wp_index_from_topic != next_light_wp_index_from_config:
                # rospy.logwarn("next_light_wp_index_from_config not matching topic, closest_wp: %s, topic: %s, config: %s", closest_wp_index, next_light_wp_index_from_topic, next_light_wp_index_from_config)
                # dist_to_topic_light = self.distance(waypoints, closest_wp_index, next_light_wp_index_from_topic)

                # dist_to_config_light = self.distance(waypoints, closest_wp_index, next_light_wp_index_from_config)

                # rospy.logwarn("topic_light x/y: %s/%s, config_light x/y: %s/%s", topic_light.pose.pose.position.x, topic_light.pose.pose.position.y, config_light.pose.pose.position.x, config_light.pose.pose.position.y)

                # rospy.logwarn("dist_to_topic_light: %s, dist_to_config_light: %s", dist_to_topic_light, dist_to_config_light)

            next_light_wp_index = next_light_wp_index_from_config
            # next_light_wp_index = next_light_wp_index_from_topic

            next_light_wp = waypoints[next_light_wp_index]

            # Ensure next_light_wp_index is in front of the car, take heading into account

            is_next_light_wp_in_front_of_the_car = self.is_next_waypoint_in_front_of_the_car(pose.pose, next_light_wp)
            # rospy.logwarn("is next_light_wp in front of the car: %s", is_next_light_wp_in_front_of_the_car)

            # if True:
            # if next_light_wp_index > closest_wp_index:
            if is_next_light_wp_in_front_of_the_car:

                dist = self.distance(waypoints, closest_wp_index, next_light_wp_index)
                # dist = -1

                # only check if we're < DISTANCE_TO_TRAFFIC_LIGHT_TO_START_CLASSIFYING meters away from next closest light
                if (dist < DISTANCE_TO_TRAFFIC_LIGHT_TO_START_CLASSIFYING):
                    config_light_state = self.get_light_state(config_light)
                    state = config_light_state
                    topic_light_state = topic_light.state
                    # state = topic_light_state

                    rospy.logwarn("closest_wp_index: %s next_wp_light_index: %s, classify_state: %s, ground_truth_state: %s, light_x: %s, light_y: %s, car_x: %s, car_y: %s, dist: %s", closest_wp_index, next_light_wp_index, config_light_state, topic_light_state, config_light.pose.pose.position.x, config_light.pose.pose.position.y, pose.pose.position.x, pose.pose.position.y, dist)

                    return next_light_wp_index, state

            # self.waypoints = None
            return -1, TrafficLight.UNKNOWN

    def get_next_point_index(self, current_pose, points):

        point_matrix = np.zeros(shape=(len(points), 2), dtype=np.float32)

        for i in range(len(points)):
            point = points[i]

            point_matrix[i, 0] = point.pose.pose.position.x
            point_matrix[i, 1] = point.pose.pose.position.y

        x_dist = point_matrix[:, 0] - current_pose.position.x
        y_dist = point_matrix[:, 1] - current_pose.position.y

        distances = np.sqrt(x_dist*x_dist + y_dist*y_dist)
        return np.argmin(distances)

    def is_next_waypoint_in_front_of_the_car(self, pose, next_wp):

        quaternion = pose.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        euler = tf.transformations.euler_from_quaternion(explicit_quat)
        car_yaw = euler[2]

        shift_x = next_wp.pose.pose.position.x - pose.position.x
        shift_y = next_wp.pose.pose.position.y - pose.position.y

        return (shift_x * math.cos(0 - car_yaw) - shift_y * math.sin(0 - car_yaw)) > 0

    def get_closest_waypoint(self, pose, waypoints):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        next_wp_index = self.get_next_point_index(pose, waypoints)
        return next_wp_index

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            wp_dist = dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            # rospy.logwarn("wp1: %s, wp2: %s, wp_dist: %s", wp1, i, wp_dist)
            dist += wp_dist
            wp1 = i
        return dist

    def distance2(self, x1, y1, x2, y2):
	    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

    def create_light(self, x, y):
        light = TrafficLight()
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        light.pose = pose
        return light

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
