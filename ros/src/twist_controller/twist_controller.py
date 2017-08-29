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

        # self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # self.pid_steering_controller = PID(.5, .0001, 4, -max_steer_angle, max_steer_angle)

        self.pid_steering_controller = PID(0.071769, 0.00411344, 0.974954, -max_steer_angle, max_steer_angle)

    def control(self, current_velocity, proposed_velocity, final_waypoints, current_pose):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        linear_velocity = proposed_velocity.linear.x
        angular_velocity = proposed_velocity.angular.z
        current_velocity = current_velocity.linear.x

        # steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        # fit polynomial to final waypoints
        x_vals = list(map(lambda wp: wp.pose.pose.position.x, final_waypoints.waypoints))

        y_vals = list(map(lambda wp: wp.pose.pose.position.y, final_waypoints.waypoints))

        current_car_x = current_pose.pose.position.x
        current_car_y = current_pose.pose.position.y

        # fit 3rd degree poly
        # z = np.polyfit(x_vals, y_vals, 3)
        # p = np.poly1d(z)
        # target_y = p(current_car_x)
        # cross_track_error = target_y - current_car_y

        # fit a spline
        tck = interpolate.splrep(x_vals, y_vals)
        target_y = interpolate.splev(current_car_x, tck, der=0)
        cross_track_error = target_y - current_car_y
        rospy.logwarn('cross_track_error: %s', cross_track_error)

        # if (abs(cross_track_error) > 0.3):
            # self.pid_steering_controller.reset()

        sample_time = .02 # ??? Match Rate in dbw_node?
        # sample_time = 1

        steer = self.pid_steering_controller.step(cross_track_error, sample_time)
        rospy.logwarn('steer: %s', steer)

        throttle = 10
        brake = 0
        # steer = 0

        # steer = angular_velocity
        # steer = -(math.pi / 16)

        return throttle, brake, steer
        # return 1., 0., 0.


