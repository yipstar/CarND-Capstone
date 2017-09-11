#! /usr/bin/python
""" 
Udacity Self-Driving Car Challenge Bag Processing
https://github.com/udacity/didi-competition/tree/master/tracklets/python

"""

from __future__ import print_function
from collections import defaultdict
import os
import sys
import math
import argparse
import functools
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyKDL as kd

from bag_topic_def import *
from bag_utils import *
from generate_tracklet import *
from scipy.spatial import kdtree
from scipy import stats

sys.path += [os.path.join(os.path.dirname(__file__), '..') +'/ros/src/tl_detector']
from traffic_light_config import config

# Bag message timestamp source
TS_SRC_PUB = 0
TS_SRC_REC = 1
TS_SRC_OBS_REC = 2

# Correction method
CORRECT_NONE = 0
CORRECT_PLANE = 1

CAP_RTK_FRONT_Z = .3323 + 1.2192
CAP_RTK_REAR_Z = .3323 + .8636

# From capture vehicle 'GPS FRONT' - 'LIDAR' in
# https://github.com/udacity/didi-competition/blob/master/mkz-description/mkz.urdf.xacro
FRONT_TO_LIDAR = [-1.0922, 0, -0.0508]

# For pedestrian capture, a different TF from mkz.urdf was used in capture. This must match
# so using that value here.
BASE_LINK_TO_LIDAR_PED = [1.9, 0., 1.6]

# CAMERA_COLS = ["timestamp", "width", "height", "frame_id", "filename"]
CAMERA_COLS = ["timestamp","filename","light","state","distance"]
# GPS_COLS = ["timestamp", "lat", "long", "alt"]
# POS_COLS = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]

#added by BinLiu 170903
_NEXT_LIGHT = -1    
_NEXT_LIGHT_STATE = 4    
_NEXT_LIGHT_DISTANCE = -1
_CURRENT_POSE = (-1,-1)
#end add 

def obs_name_from_topic(topic):
    return topic.split('/')[2]


def obs_prefix_from_topic(topic):
    words = topic.split('/')
    prefix = '_'.join(words[1:4])
    name = words[2]
    return prefix, name


def camera2dict(timestamp, msg, write_results, camera_dict):
    camera_dict["timestamp"].append(timestamp)
    if write_results:
        # camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
        # camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
        # camera_dict["frame_id"].append(msg.header.frame_id)
        camera_dict["filename"].append(write_results['filename'])
        camera_dict["light"].append(0 if _NEXT_LIGHT == -1 else _NEXT_LIGHT)        
        camera_dict["state"].append(_NEXT_LIGHT_STATE)     
        camera_dict["distance"].append(-1 if _NEXT_LIGHT_STATE == config.light_state['UNKNOW'] else _NEXT_LIGHT_DISTANCE)                       

def gps2dict(timestamp, msg, gps_dict):
    gps_dict["timestamp"].append(timestamp)
    gps_dict["lat"].append(msg.latitude)
    gps_dict["long"].append(msg.longitude)
    gps_dict["alt"].append(msg.altitude)


def pose2dict(timestamp, msg, pose_dict):
    pose_dict["timestamp"].append(timestamp)
    pose_dict["x"].append(msg.pose.position.x)
    pose_dict["y"].append(msg.pose.position.y)

def tl2dict(timestamp, tl, tl_dict):
    tl_dict["timestamp"].append(timestamp)
    # tl_dict["tx"].append(tf.translation.x)
    # tf_dict["ty"].append(tf.translation.y)
    tl_dict["nlight"].append(_NEXT_LIGHT)    
    tl_dict["state"].append(tl.state)
    tl_dict["distance"].append(-1 if tl.state == config.light_state['UNKNOW'] else _NEXT_LIGHT_DISTANCE)    
    # global _NEXT_LIGHT_STATE      
    # _NEXT_LIGHT_STATE = tl.state

def get_yaw(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])


def dict_to_vect(di):
    return kd.Vector(di['tx'], di['ty'], di['tz'])


def list_to_vect(li):
    return kd.Vector(li[0], li[1], li[2])


def vect_to_dict3(v):
    return dict(tx=v[0], ty=v[1], tz=v[2])


def vect_to_dict6(v):
    if len(v) == 6:
        return dict(tx=v[0], ty=v[1], tz=v[2], rx=v[3], ry=v[4], rz=v[5])
    else:
        return dict(tx=v[0], ty=v[1], tz=v[2], rx=0, ry=0, rz=0)


def frame_to_dict(frame, yaw_only=False):
    r, p, y = frame.M.GetRPY()
    if yaw_only:
        return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=0., ry=0., rz=y)
    return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=r, ry=p, rz=y)


def dict_to_frame(di):
    return kd.Frame(
        kd.Rotation.RPY(di['rx'], di['ry'], di['rz']),
        kd.Vector(di['tx'], di['ty'], di['tz']))


def init_df(data_dict, cols, filename, outdir=''):
    df = pd.DataFrame(data=data_dict, columns=cols)
    if len(df.index) and filename:
        df.to_csv(os.path.join(outdir, filename), index=False)
    return df


def interpolate_df(input_dfs, index_df, filter_cols=[], filename='', outdir=''):
    if not isinstance(input_dfs, list):
        input_dfs = [input_dfs]
    if not isinstance(index_df.index, pd.DatetimeIndex):
        print('Error: Camera dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for i in input_dfs:
        if len(i.index) == 0:
            print('Warning: Empty dataframe passed to interpolate, skipping.')
            return pd.DataFrame()
        i['timestamp'] = pd.to_datetime(i['timestamp'])
        i.set_index(['timestamp'], inplace=True)
        i.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right: pd.merge(
        left, right, how='outer', left_index=True, right_index=True), [index_df] + input_dfs)
    merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')

    filtered = merged.loc[index_df.index]  # back to only index' rows
    filtered.fillna(0.0, inplace=True)
    filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
    if filter_cols:
        if not 'timestamp' in filter_cols:
            filter_cols += ['timestamp']
        filtered = filtered[filter_cols]

    if len(filtered.index) and filename:
        filtered.to_csv(os.path.join(outdir, filename), header=True)
    return filtered


def obstacle_rtk_to_pose(
        cap_front,
        cap_rear,
        obs_front,
        obs_rear,
        obs_gps_to_centroid,
        front_to_velodyne,
        cap_yaw_err=0.,
        cap_pitch_err=0.):

    # calculate capture yaw in ENU frame and setup correction rotation
    cap_front_v = dict_to_vect(cap_front)
    cap_rear_v = dict_to_vect(cap_rear)
    cap_yaw = get_yaw(cap_front_v, cap_rear_v)
    cap_yaw += cap_yaw_err
    rot_cap = kd.Rotation.EulerZYX(-cap_yaw, -cap_pitch_err, 0)

    obs_rear_v = dict_to_vect(obs_rear)
    if obs_front:
        obs_front_v = dict_to_vect(obs_front)
        obs_yaw = get_yaw(obs_front_v, obs_rear_v)
        # use the front gps as the obstacle reference point if it exists as it's closers
        # to the centroid and mounting metadata seems more reliable
        cap_to_obs = obs_front_v - cap_front_v
    else:
        cap_to_obs = obs_rear_v - cap_front_v

    # transform capture car to obstacle vector into capture car velodyne lidar frame
    res = rot_cap * cap_to_obs
    res += list_to_vect(front_to_velodyne)

    # obs_gps_to_centroid is offset for front gps if it exists, otherwise rear
    obs_gps_to_centroid_v = list_to_vect(obs_gps_to_centroid)
    if obs_front:
        # if we have both front + rear RTK calculate an obstacle yaw and use it for centroid offset
        obs_rot_z = kd.Rotation.RotZ(obs_yaw - cap_yaw)
        centroid_offset = obs_rot_z * obs_gps_to_centroid_v
    else:
        # if no obstacle yaw calculation possible, treat rear RTK as centroid and offset in Z only
        obs_rot_z = kd.Rotation()
        centroid_offset = kd.Vector(0, 0, obs_gps_to_centroid_v[2])
    res += centroid_offset
    return frame_to_dict(kd.Frame(obs_rot_z, res), yaw_only=True)


def filter_outlier_points(points):
    kt = kdtree.KDTree(points)
    distances, i = kt.query(kt.data, k=9)
    z_distances = stats.zscore(np.mean(distances, axis=1))
    o_filter = abs(z_distances) < 1  # rather arbitrary
    return points[o_filter]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis /= math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def extract_metadata(md, obs_name):
    md = next(x for x in md if x['obstacle_name'] == obs_name)
    if 'gps_l' in md:
        # make old rear RTK only obstacle metadata compatible with new
        md['rear_gps_l'] = md['gps_l']
        md['rear_gps_w'] = md['gps_w']
        md['rear_gps_h'] = md['gps_h']
    return md

def process_pose_data(
        bagset,
        cap_data,
        obs_data,
        index_df,
        outdir,
):
    tracklets = []
    cap_pose_df = init_df(cap_data['base_link_pose'], POS_COLS, 'cap_pose.csv', outdir)
    cap_pose_interp = interpolate_df(
        cap_pose_df, index_df, POS_COLS, 'cap_pose_interp.csv', outdir)
    cap_pose_rec = cap_pose_interp.to_dict(orient='records')

    for obs_name, obs_pose_dict in obs_data.items():
        obs_pose_df = init_df(obs_pose_dict['pose'], POS_COLS, 'obs_pose.csv', outdir)
        obs_pose_interp = interpolate_df(
            obs_pose_df, index_df, POS_COLS, 'obs_pose_interp.csv', outdir)
        obs_pose_rec = obs_pose_interp.to_dict(orient='records')

        # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
        fig = plt.figure()
        plt.plot(
            obs_pose_interp['tx'].tolist(),
            obs_pose_interp['ty'].tolist(),
            cap_pose_interp['tx'].tolist(),
            cap_pose_interp['ty'].tolist())
        fig.savefig(os.path.join(outdir, '%s-%s-plot.png' % (bagset.name, obs_name)))
        plt.close(fig)

        # FIXME hard coded metadata, only Pedestrians currently using pose capture and there is only one person
        md = {'object_type': 'Pedestrian', 'l': 0.8, 'w': 0.8, 'h': 1.708}
        base_link_to_lidar = BASE_LINK_TO_LIDAR_PED

        obs_tracklet = Tracklet(
            object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

        def _calc_cap_to_obs(cap, obs):
            cap_frame = dict_to_frame(cap)
            obs_frame = dict_to_frame(obs)
            cap_to_obs = cap_frame.Inverse() * obs_frame
            cap_to_obs.p -= list_to_vect(base_link_to_lidar)
            cap_to_obs.p -= kd.Vector(0, 0, md['h'] / 2)
            return frame_to_dict(cap_to_obs, yaw_only=True)

        obs_tracklet.poses = [_calc_cap_to_obs(c[0], c[1]) for c in zip(cap_pose_rec, obs_pose_rec)]
        tracklets.append(obs_tracklet)
    return tracklets


#Added by BinLiu 170903
def distance(p1, p2):
    # x, y = p1.x - p2.x, p1.y - p2.y
    x, y = p1[0] - p2[0], p1[1] - p2[1]
    return math.sqrt(x*x + y*y )

def get_closest_light(pose):
    dLen = 1000000
    nLight = 0
    count = 0
    margin = 8

    for light in config.light_positions:  
        dDist = distance(pose, light) 
        if dDist < dLen:                               
            if nLight >= 0 and nLight <=3:
                if pose[0] > light[0] + margin: 
                    pass
                else:
                    dLen = dDist
                    nLight = count                                
            elif nLight >= 4 and nLight <=5:
                if pose[0] < light[0] - margin : 
                    pass
                else:
                    dLen = dDist            
                    nLight = count                                                         
            elif nLight >= 6 and nLight <=7:
                if pose[1] < light[1] - margin: 
                    pass
                else:
                    dLen = dDist             
                    nLight = count                                                        
        count +=1
    return nLight, dLen
#End add  


def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='./output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='./data',
        help='Input folder where bagfiles are located')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-t', '--ts_src', type=str, nargs='?', default='rec',
        help="""Timestamp source. 'pub'=capture node publish time, 'rec'=receiver bag record time,
        'obs_rec'=record time for obstacles topics only, pub for others. Default='pub'""")
    parser.add_argument('-c', '--correct', type=str, nargs='?', default='',
        help="""Correction method. ''=no correction, 'plane'=fit plane to RTK coords and level. Default=''""")
    parser.add_argument('--yaw_err', type=float, nargs='?', default='0.0',
        help="""Amount in degrees to compensate for RTK based yaw measurement. Default=0.0'""")
    parser.add_argument('--pitch_err', type=float, nargs='?', default='0.0',
        help="""Amount in degrees to compensate for RTK based yaw measurement. Default=0.0.""")
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-u', dest='unique_paths', action='store_true', help='Unique bag output paths')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(unique_paths=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    ts_src = TS_SRC_PUB
    if args.ts_src == 'rec':
        ts_src = TS_SRC_REC
    elif args.ts_src == 'obs_rec':
        ts_src = TS_SRC_OBS_REC
    correct = CORRECT_NONE
    if args.correct == 'plane':
        correct = CORRECT_PLANE
    yaw_err = args.yaw_err * np.pi / 180
    pitch_err = args.pitch_err * np.pi / 180
    msg_only = args.msg_only
    unique_paths = args.unique_paths
    image_bridge = ImageBridge()

    include_images = False if msg_only else True

    # filter_topics = CAMERA_TOPICS + CAP_FRONT_RTK_TOPICS + CAP_REAR_RTK_TOPICS \
    #     + CAP_FRONT_GPS_TOPICS + CAP_REAR_GPS_TOPICS

    filter_topics = CAMERA_TOPICS + [TF_LIGHTS_TOPIC]  + [CAR_POSE_TOPIC] 

    # FIXME hard coded obstacles
    # The original intent was to scan bag info for obstacles and populate dynamically in combination
    # with metadata.csv. Since obstacle names were very static, and the obstacle topic root was not consistent
    # between data releases, that didn't happen.
    obstacle_topics = []

    # For obstacles tracked via RTK messages
    # OBS_RTK_NAMES = ['obs1']
    # OBS_FRONT_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/front/gps/rtkfix' for x in OBS_RTK_NAMES]
    # OBS_REAR_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/rear/gps/rtkfix' for x in OBS_RTK_NAMES]
    # obstacle_topics += OBS_FRONT_RTK_TOPICS
    # obstacle_topics += OBS_REAR_RTK_TOPICS

    # For obstacles tracked via TF + pose messages
    # OBS_POSE_TOPICS = ['/obstacle/ped/pose']  # not under same root as other obstacles for some reason
    # obstacle_topics += OBS_POSE_TOPICS
    # filter_topics += [TF_TOPIC]  # pose based obstacles rely on TF

    # filter_topics += obstacle_topics

    bagsets = find_bagsets(indir, filter_topics=filter_topics, set_per_file=True, metadata_filename='metadata.csv')
    if not bagsets:
        print("No bags found in %s" % indir)
        exit(-1)

    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        cap_data = defaultdict(lambda: defaultdict(list))
        obs_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        outdir = os.path.join(base_outdir, bs.get_name(unique_paths))
        get_outdir(outdir)
        if include_images:
            camera_outdir = get_outdir(outdir, "camera")
            red_outdir = get_outdir(camera_outdir, "red")
            green_outdir = get_outdir(camera_outdir, "green")
            yellow_outdir = get_outdir(camera_outdir, "yellow")
            unknow_outdir = get_outdir(camera_outdir, "unknow")                                                
        bs.write_infos(outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, ts_recorded, stats):
            global _NEXT_LIGHT
            global _CURRENT_POSE
            global _NEXT_LIGHT_STATE  
            global _NEXT_LIGHT_DISTANCE             
            if topic == '/tf':
                timestamp = msg.transforms[0].header.stamp.to_nsec()
            else:
                timestamp = msg.header.stamp.to_nsec()  # default to publish timestamp in message header
            if ts_src == TS_SRC_REC: 
                timestamp = ts_recorded.to_nsec()
            elif ts_src == TS_SRC_OBS_REC and topic in obstacle_topics:
                timestamp = ts_recorded.to_nsec()

            if topic in CAMERA_TOPICS:
                write_results = {}
                if include_images:
                    if stats['msg_count'] < 15:
                        pass
                    else:
                        if _NEXT_LIGHT_DISTANCE > 195:
                            _NEXT_LIGHT_STATE = config.light_state['UNKNOW']
                        
                        if _NEXT_LIGHT_STATE == config.light_state['RED']:
                            write_results = image_bridge.write_image(red_outdir, msg,ts_recorded, fmt=img_format)
                        elif _NEXT_LIGHT_STATE == config.light_state['GREEN']:
                            write_results = image_bridge.write_image(green_outdir, msg,ts_recorded, fmt=img_format)  
                        elif _NEXT_LIGHT_STATE == config.light_state['YELLOW']:
                            write_results = image_bridge.write_image(yellow_outdir, msg,ts_recorded, fmt=img_format)                            
                        elif _NEXT_LIGHT_STATE == config.light_state['UNKNOW']:
                            write_results = image_bridge.write_image(unknow_outdir, msg,ts_recorded, fmt=img_format)                            
                        else:
                            print ("Ooooops, Wrong Light State:",_NEXT_LIGHT_STATE)

                        write_results['filename'] = os.path.relpath(write_results['filename'], outdir) 
                        camera2dict(timestamp, msg, write_results, cap_data['camera'])
                        stats['img_count'] += 1
                stats['msg_count'] += 1                

            elif topic in TF_LIGHTS_TOPIC:
                _NEXT_LIGHT, _NEXT_LIGHT_DISTANCE = get_closest_light(_CURRENT_POSE)  
                if _NEXT_LIGHT_DISTANCE > 195:
                    _NEXT_LIGHT_STATE = config.light_state['UNKNOW']

                tl2dict(timestamp, msg.lights[_NEXT_LIGHT], cap_data['light']) 
                stats['msg_count'] += 1

            elif topic in CAR_POSE_TOPIC:

                if _CURRENT_POSE[0] == msg.pose.position.x and _CURRENT_POSE[1] == msg.pose.position.y:
                    pass
                else:
                    _CURRENT_POSE = (msg.pose.position.x,msg.pose.position.y)
                    pose2dict(timestamp, msg, cap_data['pose'])
                    stats['msg_count'] += 1
            else:
                pass

        for reader in readers:
            last_img_log = 0
            last_msg_log = 0
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if last_img_log != stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0:
                    print("%d images, processed..." % stats_acc['img_count'])
                    last_img_log = stats_acc['img_count']
                    sys.stdout.flush()
                if last_msg_log != stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0:
                    print("%d messages processed..." % stats_acc['msg_count'])
                    last_msg_log = stats_acc['msg_count']
                    sys.stdout.flush()

        print("Writing done. %d images, %d messages processed." %
              (stats_acc['img_count'], stats_acc['msg_count']))
        sys.stdout.flush()

        camera_df = pd.DataFrame(data=cap_data['camera'], columns=CAMERA_COLS)
        if include_images:
            camera_df.to_csv(os.path.join(outdir, 'cap_camera.csv'), index=False)

        # if len(camera_df['timestamp']):
        #     # Interpolate samples from all used sensors to camera frame timestamps
        #     camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
        #     camera_df.set_index(['timestamp'], inplace=True)
        #     camera_df.index.rename('index', inplace=True)
        #     camera_index_df = pd.DataFrame(index=camera_df.index)

            # collection = TrackletCollection()

            # if 'front_rtk' in cap_data and 'rear_rtk' in cap_data:
            #     tracklets = process_rtk_data(
            #         bs, cap_data, obs_data, camera_index_df, outdir,
            #         correct=correct, yaw_err=yaw_err, pitch_err=pitch_err)
            #     collection.tracklets += tracklets

            # if 'base_link_pose' in cap_data:
            #     tracklets = process_pose_data(
            #         bs, cap_data, obs_data, camera_index_df, outdir)
            #     collection.tracklets += tracklets

            # if collection.tracklets:
            #     tracklet_path = os.path.join(outdir, 'tracklet_labels.xml')
            #     collection.write_xml(tracklet_path)
        else:
            print('Warning: No camera image times were found. '
                  'Skipping sensor interpolation and Tracklet generation.')


if __name__ == '__main__':
    main()
