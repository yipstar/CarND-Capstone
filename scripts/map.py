#!/usr/bin/env python

import csv
import rosbag

import matplotlib.pyplot as plt

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

def get_driven_route_points():
    bag = rosbag.Bag('bag1.bag')
    x = []
    y = []
    for topic, msg, t in bag.read_messages(topics=['/current_pose']):
        # print(msg)
        x.append(msg.pose.position.x)
        y.append(msg.pose.position.y)

    return x, y

def draw_plot():

    fig, ax = plt.subplots()

    map_x, map_y = get_map_points()

    car_x, car_y = get_driven_route_points()

    ax.plot(map_x, map_y, 'g--')
    ax.plot(car_x, car_y, 'b--')

    plt.show()


if __name__ == '__main__':
    draw_plot()
    draw_driven_route()
