#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import TwistController

from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import PoseStamped

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        max_throttle = rospy.get_param('~max_throttle', 0.8)
        max_brake = rospy.get_param('~max_brake', -0.8)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        min_speed = 0

        self.controller = TwistController(vehicle_mass, wheel_radius, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, max_throttle, max_brake, brake_deadband)

        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        self.twist_cmd = None
        self.current_velocity = None
        self.final_waypoints = None
        self.current_pose = None
        self.dbw_enabled = False

        self.previous_loop_time = rospy.get_rostime()

        self.loop()

    def twist_cb(self, msg):
        self.twist_cmd = msg.twist

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist

    def final_waypoints_cb(self, msg):
        self.final_waypoints = msg

    def pose_cb(self, msg):
        self.current_pose = msg

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data
        self.controller.reset()

    def loop(self):
        rate = rospy.Rate(10) # 50Hz

        while not rospy.is_shutdown():

            if self.current_velocity and self.twist_cmd and self.final_waypoints and self.current_pose:

                current_time = rospy.get_rostime()
                ros_duration = current_time - self.previous_loop_time
                dt = ros_duration.secs + (1e-9 * ros_duration.nsecs)
                # rospy.logwarn("dt: %s", dt)

                self.previous_loop_time = current_time

                throttle, brake, steering = self.controller.control(self.current_velocity, self.twist_cmd, self.final_waypoints, self.current_pose, dt)

                # You should only publish the control commands if dbw is enabled
                if self.dbw_enabled:
                    self.publish(throttle, brake, steering)

                rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_PERCENT
        # bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
