class SteeringController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base

    def control(self, angular_velocity):
        steer = 0
        return steer
