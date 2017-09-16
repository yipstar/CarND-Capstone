#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from geometry_msgs.msg import TwistStamped

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


        self.current_pose = None
        self.waypoints = None
        self.upcoming_red_light = None
        self.current_velocity = None

        self.time_prev = rospy.Time.now()
        self.previous_loop_time = rospy.get_rostime()

        rospy.spin()

    def current_velocity_cb(self, msg):
        self.current_velocity = msg

    def pose_cb(self, msg):
        waypoints = self.waypoints
        current_pose = msg.pose

        time_now = rospy.Time.now()
        dt = (time_now - self.time_prev).to_sec()
        # rospy.logwarn("dt: %s", dt)
        self.time_prev = time_now

        # 10 MPH convert to mps
        go_velocity = 10
        go_velocity = go_velocity * 0.44704

        # for testing faster around 
        # go_velocity = 10 # meters per second

        begin_decelerating_distance = 50.0

        current_velocity = 0
        if self.current_velocity:
            current_velocity = self.current_velocity.twist.linear.x

        stop_velocity = -1

        # assume go unless red traffic light
        target_velocity = go_velocity

        # ramp current velocity by 8/ms^2
        ramp_rate = 8.0

        if (waypoints):

            # next_wp_index = self.next_waypoint(current_pose, waypoints)
            next_wp_index = self.next_waypoint2(current_pose, waypoints)
            # rospy.logwarn("next_wp_index: %s", next_wp_index)

            final_waypoints = waypoints[next_wp_index:next_wp_index + LOOKAHEAD_WPS]

            # ranges between 200 and 100 meters depending on track location
            # final_waypoints_distance = self.distance(waypoints, next_wp_index, next_wp_index + LOOKAHEAD_WPS)
            # rospy.logwarn("final_waypoints distance: %s", final_waypoints_distance)

            dist = None
            wp_offset = 0 # stop 2 waypoints ahead of the light
            if self.upcoming_red_light and self.upcoming_red_light != -1:

                stop_index = self.upcoming_red_light - wp_offset
                dist = self.distance(waypoints, next_wp_index, stop_index)

                # rospy.logwarn("upcoming_red_light_wp: %s, dist: %s", stop_index, dist)
                if stop_index > next_wp_index and dist < begin_decelerating_distance:
                    target_velocity = stop_velocity

            wp_vel = current_velocity
            for i in range(len(final_waypoints)):

                ramped_vel = self.ramped_velocity(wp_vel, target_velocity, 1.0, ramp_rate)

                self.set_waypoint_velocity(final_waypoints, i, ramped_vel)
                wp_vel = ramped_vel

            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time(0)
            lane.waypoints = final_waypoints
            self.final_waypoints_pub.publish(lane)

    def ramped_velocity(self, v_prev, v_target, dt, ramp_rate):
        step = ramp_rate * dt
        sign = 1.0 if (v_target > v_prev) else -1.0
        error = math.fabs(v_target - v_prev)
        if error < step:
            # we can get there within this time step, we're done.
            return v_target
        else:
            return v_prev + sign * step # take a step toward the target

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.upcoming_red_light = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def next_waypoint(self, current_pose, waypoints):
        closest_len = 1000000; # large number
        closest_waypoint_index = 0

        distances = []

        for i in range(len(waypoints)):
            wp = waypoints[i]

            car_x = current_pose.position.x
            car_y = current_pose.position.y

            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y

            dist = self.distance2(car_x, car_y, wp_x, wp_y)
            distances.append(dist)

        return np.argmin(distances)

    def next_waypoint2(self, current_pose, waypoints):

        wp_matrix = np.zeros(shape=(len(waypoints), 2), dtype=np.float32)

        for i in range(len(waypoints)):
            wp = waypoints[i]

            wp_matrix[i, 0] = wp.pose.pose.position.x
            wp_matrix[i, 1] = wp.pose.pose.position.y

        x_dist = wp_matrix[:, 0] - current_pose.position.x
        y_dist = wp_matrix[:, 1] - current_pose.position.y

        distances = np.sqrt(x_dist*x_dist + y_dist*y_dist)
        return np.argmin(distances)

    def distance2(self, x1, y1, x2, y2):
	    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
