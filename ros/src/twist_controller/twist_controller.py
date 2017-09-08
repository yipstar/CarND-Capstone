import math
import rospy

from steering_controller import SteeringController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class TwistController(object):

    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):

        self.wheel_base = wheel_base

        self.steering_controller = SteeringController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.throttle_controller = PID(1.0, 0.01, 0.5, 0.0, 1.0)

    def rad2deg(self, radians):
	    degrees = 180 * radians / math.pi
	    return degrees

    def deg2rad(self, degrees):
	    radians = math.pi * degrees / 180
	    return radians

    def control(self, current_velocity, twist_cmd, final_waypoints, current_pose, dt):

        proposed_linear_velocity = twist_cmd.linear.x
        proposed_angular_velocity = twist_cmd.angular.z
        current_linear_velocity = current_velocity.linear.x

        steer = self.steering_controller.control(proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, final_waypoints, current_pose, dt)

        # throttle = 0.
        brake = 0.

        velocity_cte = proposed_linear_velocity - current_linear_velocity
        throttle = self.throttle_controller.step(velocity_cte, dt)
        rospy.logwarn("velocity_cte: %s, throttle: %s", velocity_cte, throttle)

        # velocity_cte = proposed_linear_velocity - current_linear_velocity
        # if(current_linear_velocity < proposed_linear_velocity):
        #   throttle = min(1.0, velocity_cte)

        # if(current_linear_velocity > proposed_linear_velocity):
        #   brake = 1000.

        # throttle = 1.0
        # throttle = 0.5

        rospy.logwarn('plv: %s pav: %s cv: %s', proposed_linear_velocity, proposed_angular_velocity,  current_linear_velocity)

        rospy.logwarn('throttle: %s brake: %s steer: %s', throttle, brake,  steer)

        return throttle, brake, steer


