import math
import rospy

from steering_controller import SteeringController
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class TwistController(object):

    def __init__(self, vehicle_mass, wheel_radius, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, max_throttle, max_brake, brake_deadband):

        self.vehicle_mass = vehicle_mass
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.brake_deadband = brake_deadband

        self.steering_controller = SteeringController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.throttle_controller = PID(1.0, 0.01, 0.5, -max_throttle, max_throttle)

        self.previous_linear_velocity = 0.

        # Assumes we are braking from 10 m/s velocity down to 0 m/s in 5 seconds to acheive average deceleration of 2 m/s^2.
        # average_deceleration = 2
        # self.brake_torque = self.vehicle_mass * average_deceleration * self.wheel_radius

    def reset(self):
        self.steering_controller.reset()
        self.throttle_controller.reset()

    def control(self, current_velocity, twist_cmd, final_waypoints, current_pose, dt):
        accel = 0.

        proposed_linear_velocity = twist_cmd.linear.x
        proposed_angular_velocity = twist_cmd.angular.z
        current_linear_velocity = current_velocity.linear.x

        if self.previous_linear_velocity and current_linear_velocity:
            accel = (current_linear_velocity - self.previous_linear_velocity) * dt
            # rospy.logwarn("accel: %s", accel)

        self.previous_linear_velocity = current_linear_velocity

        steer = self.steering_controller.control(proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, final_waypoints, current_pose, dt)

        velocity_error = proposed_linear_velocity - current_linear_velocity

        throttle = self.throttle_controller.step(velocity_error, dt)
        rospy.logwarn("velocity error: %s throttle: %s", velocity_error, throttle)

        # We should be about to stop or stopped, stay braking
        if proposed_linear_velocity == 0.0:
            brake = 80.0
            throttle = 0.0

        # Probably breaking and not crusing so perform brake
        elif velocity_error <= -0.6:

            brake = abs(throttle)
            if brake > self.max_brake:
                brake = self.max_brake
            brake = brake * 100.0
            throttle = 0.0

        # Crusing
        else:
            if throttle < 0:
                throttle = 0
            brake = 0

        # rospy.logwarn('throttle: %s brake: %s steer: %s accel: %s', throttle, brake, steer, accel)

        return throttle, brake, steer

    def throttle_percentage_for_velocity(self, target_velocity):
        return target_velocity / 150.0

    def rad2deg(self, radians):
	    degrees = 180 * radians / math.pi
	    return degrees

    def deg2rad(self, degrees):
	    radians = math.pi * degrees / 180
	    return radian



