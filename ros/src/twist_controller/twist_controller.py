import math
import numpy as np
import rospy
from scipy import interpolate
from scipy.interpolate import UnivariateSpline

from geometry_msgs.msg import PoseStamped

from yaw_controller import YawController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class TwistController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):

        # TODO: Implement

        # self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # self.pid_steering_controller = PID(.5, .0001, 4, -max_steer_angle, max_steer_angle)

        # max_steer_angle = 3.0

        self.pid_steering_controller = PID(0.071769, 0.00411344, 0.974954, -max_steer_angle, max_steer_angle)

        # self.pid_steering_controller = PID(.2, .0001, .5, -max_steer_angle, max_steer_angle)

        # self.pid_steering_controller = PID(.5, .005, 1, -max_steer_angle, max_steer_angle)

        # self.pid_steering_controller = PID(.125, 0.0005, .2, -max_steer_angle, max_steer_angle)

        self.change_lane_steering_controller = PID(.2, 0, 0, -2.0, 2.0)

        self.cycle = 0

        self.spline_pub = rospy.Publisher('/spline_pose', PoseStamped, queue_size=1)

        self.poly_pub = rospy.Publisher('/poly_pose', PoseStamped, queue_size=1)

    # def get_freenet(x, y, theta, )

    def control(self, current_velocity, proposed_velocity, final_waypoints, curve_ref_waypoints, current_pose):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        linear_velocity = proposed_velocity.linear.x
        angular_velocity = proposed_velocity.angular.z
        current_velocity = current_velocity.linear.x

        # steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        # fit polynomial to final waypoints
        # x_vals = list(map(lambda wp: wp.pose.pose.position.x, final_waypoints.waypoints))

        # y_vals = list(map(lambda wp: wp.pose.pose.position.y, final_waypoints.waypoints))

        # fit polynomial to curve_ref_waypoints
        x_vals = list(map(lambda wp: wp.pose.pose.position.x, curve_ref_waypoints.waypoints))

        y_vals = list(map(lambda wp: wp.pose.pose.position.y, curve_ref_waypoints.waypoints))

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

        # sample_time = .02 # ??? Match Rate in dbw_node?
        sample_time = 1

        # clear the integral component every .5 seconds
        if (self.cycle % 10 == 1):
            self.pid_steering_controller.reset()

        # TODO: if cross_track_error > 7.0 then we have a lane change

        # if (abs(cross_track_error) > 15.0):
        #     rospy.logwarn('changing lane')
        #     steer = self.change_lane_steering_controller.step(cross_track_error, sample_time)

        #     # rospy.signal_shutdown("Possible Lane Change")
        # else:

        # if (abs(cross_track_error) > 15.0):
        #     rospy.signal_shutdown("Possible Lane Change")

        steer = self.pid_steering_controller.step(cross_track_error, sample_time)

        if (abs(cross_track_error) > 3.0):
            rospy.logwarn('cross_track_error: %s', cross_track_error)
            rospy.logwarn('steer: %s', steer)
            rospy.logwarn('car_y: %s', current_car_y)
            rospy.logwarn('target_y: %s', target_y)

        # if (abs(cross_track_error) > 9.0):


        throttle = 10
        brake = 0
        # steer = 0

        # steer = angular_velocity
        # steer = -(math.pi / 16)

        self.cycle += 1

        return throttle, brake, steer
        # return 1., 0., 0.


