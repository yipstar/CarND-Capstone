import math
import numpy as np
import rospy
from scipy import interpolate

from yaw_controller import YawController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class TwistController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):

        # TODO: Implement

        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    def rad2deg(self, radians):
	degrees = 180 * radians / math.pi
	return degrees

    def deg2rad(self, degrees):
	radians = math.pi * degrees / 180
	return radians

    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, current_angular_velocity, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # calculate predictive steer component
        steer = self.yaw_controller.get_steering(proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity)

        # TODO Calculate steering_cte

        throttle = 0.
        brake = 0.
        velocity_cte = proposed_linear_velocity - current_linear_velocity
        if(current_linear_velocity < proposed_linear_velocity):
          throttle = min(1.0, velocity_cte)  

        if(current_linear_velocity > proposed_linear_velocity):
          brake = 1000.

        rospy.logwarn('plv: %s pav: %s cv: %s', proposed_linear_velocity, proposed_angular_velocity,  current_linear_velocity)
        rospy.logwarn('throttle: %s brake: %s steer: %s', throttle, brake,  steer)

        return throttle, brake, steer


