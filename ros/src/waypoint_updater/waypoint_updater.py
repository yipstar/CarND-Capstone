#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

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

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.curve_ref_waypoints_pub = rospy.Publisher('curve_ref_waypoints', Lane, queue_size=1)

        # self.spline_pub = rospy.Publisher('spline', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.current_pose = None

        self.waypoints = None

        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        rospy.loginfo('current_pose x, y: %s, %s', msg.pose.position.x, msg.pose.position.y)
        current_pose = msg.pose

        if (self.waypoints):
            next_wp_index = self.next_waypoint(msg.pose.position.x, msg.pose.position.y, 0, self.waypoints)

            # rospy.loginfo('next_wp_index %s: ', next_wp_index)
            waypoints = self.waypoints[next_wp_index:next_wp_index + LOOKAHEAD_WPS]

            curve_ref_waypoints = self.waypoints[next_wp_index - 10: next_wp_index + LOOKAHEAD_WPS]

            last_wp = self.waypoints[next_wp_index - 1]

            for i in range(len(waypoints)):
                self.set_waypoint_velocity(waypoints, i, 10)
                next_wp = waypoints[i]
                rospy.loginfo('next waypoint x, y, v: %s, %s, %s', next_wp.pose.pose.position.x, next_wp.pose.pose.position.y, next_wp.twist.twist.linear.x)

            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time(0)
            lane.waypoints = waypoints
            self.final_waypoints_pub.publish(lane)

            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time(0)
            lane.waypoints = waypoints
            self.curve_ref_waypoints_pub.publish(lane)


    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # rospy.loginfo('waypoints size: %s', len(waypoints.waypoints))
        self.waypoints = waypoints.waypoints
        # pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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

    def next_waypoint(self, x, y, theta, waypoints):
        closest_waypoint_index = self.closest_waypoint(x, y, waypoints)

        wp = waypoints[closest_waypoint_index]

        heading = math.atan2( (wp.pose.pose.position.y - y), (wp.pose.pose.position.x - x))
        angle = abs(theta - heading)

        if (angle > math.pi / 4):
            closest_waypoint_index += 1

        return closest_waypoint_index

    def closest_waypoint(self, x, y, waypoints):
        closest_len = 1000000; # large number
        closest_waypoint_index = 0

        for i in range(len(waypoints)):
            wp = waypoints[i]
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y

            dist = self.distance2(x, y, wp_x, wp_y)
            if (dist < closest_len):
                closest_len = dist
                closest_waypoint_index = i

        return closest_waypoint_index

    def distance2(self, x1, y1, x2, y2):
	    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
