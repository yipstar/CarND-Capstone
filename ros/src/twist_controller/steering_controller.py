class SteeringController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base

    def control(self, angular_velocity):
        steer = 0

        # quaternion = current_pose.pose.orientation
        # explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        # euler = tf.transformations.euler_from_quaternion(explicit_quat)
        # yaw = euler[2]
        # rospy.logwarn("current yaw before steering: %s", yaw)

        # steer = -(yaw + (angular_velocity * dt))

        # rospy.logwarn("twist angular_velocity: %s", angular_velocity)

        # L = self.wheel_base
        # steer = yaw + linear_velocity / L * math.tan(angular_velocity) * dt
        # state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
        # steer = theta

        # wheel_base = L
        # omega is orientation
        # V is angual velocity
        # w looking sign is heading or yaw

        return steer
