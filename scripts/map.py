#!/usr/bin/env python

import csv
import rosbag

import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

CSV_HEADER = ['x', 'y', 'z', 'yaw']

def load_waypoints(fname):
    x = []
    y = []
    with open(fname) as wfile:
        reader = csv.DictReader(wfile, CSV_HEADER)
        for wp in reader:
            x.append(float(wp['x']))
            y.append(float(wp['y']))

    return x,y


def get_map_points():
    path = "../data/wp_yaw_const.txt"
    x, y = load_waypoints(path)
    return x, y

def get_driven_route_points(bagname):
    bag = rosbag.Bag(bagname)
    x = []
    y = []
    for topic, msg, t in bag.read_messages(topics=['/current_pose']):
        # print(msg)
        x.append(msg.pose.position.x)
        y.append(msg.pose.position.y)

    return x, y

def get_spline_points(bagname):
    bag = rosbag.Bag(bagname)
    x = []
    y = []
    for topic, msg, t in bag.read_messages(topics=['/spline_pose']):
        # print(msg)
        x.append(msg.pose.position.x)
        y.append(msg.pose.position.y)

    return x, y

def get_poly_points(bagname):
    bag = rosbag.Bag(bagname)
    x = []
    y = []
    for topic, msg, t in bag.read_messages(topics=['/poly_pose']):
        # print(msg)
        x.append(msg.pose.position.x)
        y.append(msg.pose.position.y)

    return x, y

def get_test_spline(map_x, map_y):
    x = []
    y = []
    spl = UnivariateSpline(map_x, map_y)
    return x, y


def draw_plot():

    bagname = 'bag6.bag'

    fig, ax = plt.subplots()

    map_x, map_y = get_map_points()

    car_x, car_y = get_driven_route_points(bagname)

    spline_x, spline_y = get_spline_points(bagname)

    # test_spline_x, test_spline_y = get_test_spline(map_x, map_y)

    poly_x, poly_y = get_poly_points(bagname)

    print('car_x points: %s', len(car_x))
    print('spline_x points: %s', len(spline_x))
    print('poly_x points: %s', len(poly_x))

    ax.plot(map_x, map_y, 'g--')
    # ax.plot(car_x, car_y, 'b--')

    ax.plot(spline_x, spline_y, 'r--')
    # ax.plot(poly_x, poly_y, 'r--')

    plt.show()


if __name__ == '__main__':
    draw_plot()
    # draw_spline()
