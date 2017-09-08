import math
import numpy as np
import rospy
import tf
from scipy.interpolate import CubicSpline

from geometry_msgs.msg import PoseStamped
from yaw_controller import YawController
from pid import PID

class SteeringController(object):

    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base

        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.pid_steering_controller = PID(1.0, 0.001, 0.5, -max_steer_angle, max_steer_angle)

        self.pi_steering_controller = PID(1.0, 0.001, 0, -max_steer_angle, max_steer_angle)


        self.poly_pub = rospy.Publisher('/poly_pose', PoseStamped, queue_size=1)

    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, final_waypoints, current_pose, dt):

        # calculate predictive steer component
        yaw_steer = -self.yaw_controller.get_steering(proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity)

        # next fit polynomial to waypoints, shift, rotate, and calc cte
        # https://www.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point

        # fit polynomial to final waypoints
        current_car_x = current_pose.pose.position.x
        current_car_y = current_pose.pose.position.y

        # num_waypoints_for_curvature = 10
        num_waypoints_for_curvature = len(final_waypoints.waypoints)

        # use the first next waypoint as the origin to rotate around
        origin = final_waypoints.waypoints[0].pose.pose.position

        # first shift waypoints to origin
        shifted_points = []

        # TODO: vectorize this using numpy
        for i in range(num_waypoints_for_curvature):
            shift_x = final_waypoints.waypoints[i].pose.pose.position.x - origin.x
            shift_y = final_waypoints.waypoints[i].pose.pose.position.y - origin.y

            shifted_points.append([shift_x, shift_y])

        shifted_points = np.array(shifted_points)

        current_yaw = tf.transformations.euler_from_quaternion([current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w])[2]

        theta = current_yaw

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        rotated_points = np.dot(shifted_points, rotation_matrix)

        # fit 2rd degree poly
        coeff = np.polyfit(rotated_points[:, 0], rotated_points[:, 1], 2)
        coeff2 = np.polyfit(shifted_points[:, 0], shifted_points[:, 1], 2)

        # now shift and rotate our current pose in the same way
        current_x_shifted = current_car_x - origin.x
        current_y_shifted = current_car_y - origin.y

        # now rotate with same rotation_matrix
        rotated_car_point = np.dot(np.array([current_x_shifted, current_y_shifted]), rotation_matrix)

        # calc cte
        target_y = np.polyval(coeff, rotated_car_point[0])
        actual_y = rotated_car_point[1]
        cte = -(actual_y - target_y)

        # cte2 is cte without rotating the waypoints, use the debug logging
        # to see how they deviate during track curvature
        target_y2 = np.polyval(coeff2, current_x_shifted)
        actual_y2 = current_y_shifted
        cte2 = -(actual_y2 - target_y2)

        # timestamp = rospy.Time(0)
        # msg = PoseStamped()
        # msg.header.frame_id = '/world'
        # msg.header.stamp = timestamp
        # msg.pose.position.x = current_car_x
        # msg.pose.position.y = target_y
        # msg.pose.position.z = cte
        # self.poly_pub.publish(msg)

        pid_steer = self.pid_steering_controller.step(cte, dt)
        pi_steer = self.pi_steering_controller.step(cte, dt)

        rospy.logwarn("CTE: %s, yaw_steer: %s, pid_steer: %s, pi_steer: %s", cte, yaw_steer, pid_steer, pi_steer)
        # rospy.logwarn("CTE2: %s", cte2)

        # steer = yaw_steer + pi_steer
        steer = pid_steer
        # steer = pi_steer
        # steer = pid_steer - yaw_steer
        # steer = yaw_steer

        return steer
