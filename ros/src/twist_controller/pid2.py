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

        self.int_val = self.last_int_val = self.last_error = 0.

        self.i_error = 0

    def reset(self):
        self.int_val = 0.0
        self.last_int_val = 0.0

    def step(self, error, sample_time):

        d_error = (error - self.last_error)
        p_error = error
        i_error = self.i_error + error
        self.i_error = i_error

        rospy.logwarn('p_error: %s', p_error)
        rospy.logwarn('d_error: %s', d_error)
        rospy.logwarn('i_error: %s', i_error)

        y = self.kp * p_error + self.ki * i_error + self.kd * d_error;
        val = max(self.min, min(y, self.max))

        rospy.logwarn('val: %s', val)

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = i_error

        # self.last_error = error

        rospy.logwarn('val: %s', val)

        return val
