import rospy

MIN_NUM = float('-inf')
MAX_NUM = float('inf')

class PID(object):

    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.min = mn
        self.max = mx

        self.i_error = 0.0
        self.last_error = 0.0

    def reset(self):

        self.last_error = 0.0
        self.i_error = 0.0

    def step(self, error, sample_time):

        self.i_error += error * sample_time

        d_error = (error - self.last_error) / sample_time

        self.last_error = error

        y = (self.kp * error) + (self.ki * self.i_error) + (self.kd * d_error)

        # rospy.logwarn("y: %s", y)

        return max(self.min, min(y, self.max))
