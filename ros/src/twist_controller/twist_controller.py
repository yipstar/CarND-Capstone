import math
import numpy as np
import rospy
import tf

from scipy import interpolate
from scipy.interpolate import UnivariateSpline

from geometry_msgs.msg import PoseStamped

from yaw_controller import YawController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class TwistController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):

        self.wheel_base = wheel_base

        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.pid_steering_controller = PID(0.071769, 0.00411344, 0.974954, -max_steer_angle, max_steer_angle)

        # self.pid_steering_controller = PID(.2, .0001, .5, -max_steer_angle, max_steer_angle)

        # self.pid_steering_controller = PID(.5, .005, 1, -max_steer_angle, max_steer_angle)

        self.cycle = 0

        self.spline_pub = rospy.Publisher('/spline_pose', PoseStamped, queue_size=1)

        self.poly_pub = rospy.Publisher('/poly_pose', PoseStamped, queue_size=1)

        self.last_theta = None


    def rad2deg(self, radians):
	degrees = 180 * radians / math.pi
	return degrees

    def deg2rad(self, degrees):
	radians = math.pi * degrees / 180
	return radians

    # def get_freenet(x, y, theta, )

    def control(self, current_velocity, proposed_velocity, final_waypoints, curve_ref_waypoints, current_pose, dt):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        linear_velocity = proposed_velocity.linear.x
        proposed_linear_velocity = linear_velocity

        angular_velocity = proposed_velocity.angular.z
        proposed_angular_velocity = angular_velocity

        current_velocity = current_velocity.linear.x
        current_linear_velocity = current_velocity

        # calculate predictive steer component

        steer = -self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        # fit polynomial to final waypoints
        x_vals = list(map(lambda wp: wp.pose.pose.position.x, final_waypoints.waypoints))

        y_vals = list(map(lambda wp: wp.pose.pose.position.y, final_waypoints.waypoints))

        # fit polynomial to curve_ref_waypoints
        # x_vals = list(map(lambda wp: wp.pose.pose.position.x, curve_ref_waypoints.waypoints))

        # y_vals = list(map(lambda wp: wp.pose.pose.position.y, curve_ref_waypoints.waypoints))

        current_car_x = current_pose.pose.position.x
        current_car_y = current_pose.pose.position.y

        timestamp = rospy.Time(0)

        # fit 3rd degree poly
        # try 2nd degree
        z = np.polyfit(x_vals[0:50], y_vals[0:50], 2)
        p = np.poly1d(z)
        target_y = p(current_car_x)
        cross_track_error = target_y - current_car_y

        msg = PoseStamped()
        msg.header.frame_id = '/world'
        msg.header.stamp = timestamp
        msg.pose.position.x = current_car_x
        msg.pose.position.y = target_y
        msg.pose.position.z = cross_track_error # just to use for plotting
        self.poly_pub.publish(msg)

        # fit a spline
        spl = UnivariateSpline(x_vals, y_vals)
        spl.set_smoothing_factor(0.1)
        target_y2 = spl(current_car_x)
        cross_track_error2 = target_y2 - current_car_y

        msg = PoseStamped()
        msg.header.frame_id = '/world'
        msg.header.stamp = timestamp
        msg.pose.position.x = current_car_x
        msg.pose.position.y = target_y2
        msg.pose.position.z = cross_track_error2 # just to use for plotting
        self.spline_pub.publish(msg)

        # clear the integral component every .5 seconds
        # if (self.cycle % 10 == 1):
        #     self.pid_steering_controller.reset()

        # if cross_track_error > 7.0
        # rospy.signal_shutdown("High CTE")

        # TODO Calculate steering_cte
        # steer = self.pid_steering_controller.step(cross_track_error, sample_time)

        throttle = 0.
        brake = 0.
        velocity_cte = proposed_linear_velocity - current_linear_velocity
        if(current_linear_velocity < proposed_linear_velocity):
          throttle = min(1.0, velocity_cte)

        if(current_linear_velocity > proposed_linear_velocity):
          brake = 1000.

        self.cycle += 1

        throttle = 1.0

        rospy.logwarn('plv: %s pav: %s cv: %s', proposed_linear_velocity, proposed_angular_velocity,  current_linear_velocity)
        rospy.logwarn('throttle: %s brake: %s steer: %s', throttle, brake,  steer)

        return throttle, brake, steer


