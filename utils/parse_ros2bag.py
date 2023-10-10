from reprefect import task, flow, describe

import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import cv2
import numpy as np
import os
from IPython import embed
import pylas
import refile
import subprocess
import time
import shutil
import nori2 as nori
import pyproj
from tqdm import tqdm
import rrun
import pickle
import json
from rslidar_decoder import RSLidarDecoder
import io
from functools import partial
from restore.archive import Nori2RemoteArchiveWriter
import getpass
'''
Topic information: Topic: /cam_10/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 675 | Serialization Format: cdr
                   Topic: /cam_11/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 675 | Serialization Format: cdr
                   Topic: /cam_15/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 675 | Serialization Format: cdr
                   Topic: /cam_2/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 675 | Serialization Format: cdr
                   Topic: /cam_5/image_raw/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 675 | Serialization Format: cdr
                   Topic: /left/rslidar_packets | Type: rslidar_msg/msg/RslidarScan | Count: 675 | Serialization Format: cdr
                   Topic: /left/rslidar_packets_difop | Type: rslidar_msg/msg/RslidarPacket | Count: 675 | Serialization Format: cdr
                   Topic: /right/rslidar_packets | Type: rslidar_msg/msg/RslidarScan | Count: 675 | Serialization Format: cdr
                   Topic: /right/rslidar_packets_difop | Type: rslidar_msg/msg/RslidarPacket | Count: 675 | Serialization Format: cdr
                   Topic: /rslidar_packets | Type: rslidar_msg/msg/RslidarScan | Count: 675 | Serialization Format: cdr
                   Topic: /rslidar_packets_difop | Type: rslidar_msg/msg/RslidarPacket | Count: 68 | Serialization Format: cdr
                   Topic: /sensor/corr_imu | Type: devastator_localization_msgs/msg/CorrectedImu | Count: 8410 | Serialization Format: cdr
                   Topic: /sensor/gps | Type: devastator_localization_msgs/msg/GnssBestPose | Count: 68 | Serialization Format: cdr
                   Topic: /sensor/ins_pose | Type: devastator_localization_msgs/msg/Ins | Count: 8384 | Serialization Format: cdr
                   Topic: /sensor/raw_imu | Type: devastator_localization_msgs/msg/RawImu | Count: 8391 | Serialization Format: cdr
                   Topic: /sensor/std_odom | Type: nav_msgs/msg/Odometry | Count: 8387 | Serialization Format: cdr
                   Topic: /sensor/vehicle_report | Type: vehicle_info_msgs/msg/VehicleInfo | Count: 3375 | Serialization Format: cdr
'''

# 支持的类别
# topic_name : 存储的位置
topictype2output = {
    'sensor_msgs/msg/CompressedImage': 'image',
    'rslidar_msg/msg/RslidarScan': 'pointcloud',
    'devastator_localization_msgs/msg/CorrectedImu': 'corr_imu',
    'devastator_localization_msgs/msg/GnssBestPose': 'gps',
    'devastator_localization_msgs/msg/Ins': 'ins',
    'devastator_localization_msgs/msg/RawImu': 'rawimu',
    'nav_msgs/msg/Odometry': 'odom',
    'devastator_perception_msgs/msg/RadarObjectArray': 'radar',
    'devastator_control_msgs/msg/VehicleReportInfo': 'vehicle_report',
}


def callback(result, idx, timestamp, nid, e):
    if e != None:
        print(e)
        return
    result.append((idx, timestamp, nid))


def get_point(p):
    return {
        "x": p.x,
        "y": p.y,
        "z": p.z,
    }


def check_s3_nori_num(conn, output_folder, topic_id, topic_name, topic_type,
                      output_type_folder):
    # 查看包里的对应msg数量
    sql = "select count(*) from messages where topic_id={}".format(topic_id)
    rowcount = conn.execute(sql).fetchall()[0][0]
    if topic_name == '/rslidar_packets':
        output_name = 'middle_lidar'
    elif topic_name == '/left/rslidar_packets':
        output_name = 'left_lidar'
    elif topic_name == '/right/rslidar_packets':
        output_name = 'right_lidar'
    else:
        output_name = topic_name.replace('/', '#')

    if topic_type == 'rslidar_msg/msg/RslidarScan':
        norilist_name = 'pointcloud_norilist.txt'
    elif topic_type == 'sensor_msgs/msg/CompressedImage':
        norilist_name = 'image_norilist.txt'
    else:
        norilist_name = 'timestamplist.txt'
    output_norilist = os.path.join(output_type_folder, output_name,
                                   norilist_name)
    # 不存在*txt文件，这一topic没有解完，删掉
    topic_folder = os.path.join(output_type_folder, output_name)
    if not refile.smart_exists(output_norilist):
        # cmd = "aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 rm {} --recursive".format(
        #     os.path.join(output_type_folder, output_name))
        # print('not exists. {}'.format(output_norilist))
        # print(cmd)
        # rm_nori = subprocess.run(cmd, shell=True)
        refile.smart_remove(topic_folder, missing_ok=True)
        print('path doesnot exist: {}'.format(output_norilist))
        return False
    elif topic_type == 'rslidar_msg/msg/RslidarScan':
        nori_path = os.path.join(output_type_folder, 'pointcloud.nori')
        if not refile.smart_exists(nori_path) or len(
                refile.s3_listdir(nori_path)) % 3 != 0 or len(
                    refile.s3_listdir(nori_path)) < 3:
            print('bad nori: {}'.format(nori_path))
            # cmd = "aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 rm {} --recursive".format(
            #     os.path.join(output_type_folder, output_name))
            # print(cmd)
            # rm_nori = subprocess.run(cmd, shell=True)
            refile.smart_remove(topic_folder, missing_ok=True)
            print('path doesnot exist or wrong:  {}'.format(nori_path))
            return False
    elif topic_type == 'sensor_msgs/msg/CompressedImage':
        nori_path = os.path.join(output_type_folder, 'image.nori')
        if not refile.smart_exists(nori_path) or len(
                refile.s3_listdir(nori_path)) % 3 != 0 or len(
                    refile.s3_listdir(nori_path)) < 3:
            print('bad nori: {}'.format(nori_path))
            # cmd = "aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 rm {} --recursive".format(
            #     os.path.join(output_type_folder, output_name))
            # print(cmd)
            # rm_nori = subprocess.run(cmd, shell=True)
            refile.smart_remove(topic_folder, missing_ok=True)
            print('path doesnot exist or wrong: {}'.format(nori_path))
            return False

    else:
        awscount = len(refile.smart_open(output_norilist).readlines())
        if rowcount >= awscount + 1:
            print(output_norilist, rowcount, awscount)
            # cmd = "aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 rm {} --recursive".format(
            #     os.path.join(output_type_folder, output_name))
            # print(cmd)
            # rm_nori = subprocess.run(cmd, shell=True)
            refile.smart_remove(topic_folder, missing_ok=True)
            print('path not finish: {}'.format(output_norilist))
            return False
    print('return True: {}'.format(output_norilist))
    return True


def deal_with_image(topic_name, output_type_folder, topic_id, conn, topic_type,
                    bag_path):
    output_name = topic_name.replace("/", "#")
    output_image_topic_path = os.path.join(output_type_folder, output_name)
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)
    nori_path = os.path.join(bag_path, 'image.nori')
    print(nori_path)
    image_norilist = os.path.join(output_image_topic_path,
                                  'image_norilist.txt')
    print(image_norilist)
    result = []
    cursor_list = list(cursor)
    if not cursor_list:
        return

    nw = Nori2RemoteArchiveWriter(file_uri=nori_path,
                                  append_mode=True,
                                  remote_writer_mode=2)
    keys = []
    contents = []
    # nw = nori.remotewriteopen(nori_path, mode=2)
    for idx, row in tqdm(enumerate(cursor_list)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        keys.append(timestamp)
        contents.append(msg.data.tostring())
        # nw.async_put(partial(callback, result, idx, timestamp),
        #              msg.data.tostring(),
        #              filename=timestamp)
    # nw.join()
    # result = sorted(result, key=lambda x: x[0])

    results = nw.extend_contents(keys=keys, contents=contents)
    nw.close()
    assert len(results) == len(keys) == len(contents)
    with refile.smart_open(image_norilist, "w") as w:
        for timestamp, nori_id in zip(keys, results):
            w.write(timestamp + '#' + nori_id + '\n')


def deal_with_rawimu(topic_name, output_type_folder, topic_id, conn,
                     topic_type):
    output_name = topic_name.replace("/", "#")
    output_rawimu_topic_path = os.path.join(output_type_folder, output_name)
    count = 10
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)

    rawimu_list = []
    for row in tqdm(list(cursor)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        #             # 存成二进制
        rawimu_dict = {
            "timestamp": timestamp,
            "measurement_span": msg.measurement_span,
            "linear_acceleration": get_point(msg.linear_acceleration),
            "angular_velocity": get_point(msg.angular_velocity),
        }
        rawimu_list.append(rawimu_dict)
    data_path = os.path.join(output_rawimu_topic_path, 'rawimu.json')
    with refile.smart_open(data_path, "w") as w:
        json.dump(rawimu_list, w)
    timestamplist = os.path.join(output_rawimu_topic_path, 'timestamplist.txt')
    print(data_path)
    print(timestamplist)
    with refile.smart_open(timestamplist, "w") as w:
        for i in rawimu_list:
            w.write(i["timestamp"] + "\n")


def deal_with_vehiclereport(topic_name, output_type_folder, topic_id, conn,
                            topic_type):
    output_name = topic_name.replace("/", "#")
    output_vehiclereport_topic_path = os.path.join(output_type_folder,
                                                   output_name)
    count = 10
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)
    vehiclereport_list = []
    for row in tqdm(list(cursor)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        #             # 存成二进制
        vehiclereport_dict = {
            "timestamp": timestamp,
            "gear_en": msg.gear_en,
            "gear": msg.gear,
            "speed": msg.speed,
            "long_acc": msg.long_acc,
            "late_acc": msg.late_acc,
            "long_flt": msg.long_flt,
            "long_en": msg.long_en,
            "late_en": msg.late_en,
            "steer_angle": msg.steer_angle,
            "yaw_rate": msg.yaw_rate,
            "late_flt": msg.late_flt,
            "turn_signal_status": msg.turn_signal_status,
            "wiper_status": msg.wiper_status,
            "door_fl": msg.door_fl,
            "door_fr": msg.door_fr,
            "door_rl": msg.door_rl,
            "door_rr": msg.door_rr,
            "wheelspeed_fl": msg.wheelspeed_fl,
            "wheelspeed_fr": msg.wheelspeed_fr,
            "wheelspeed_rl": msg.wheelspeed_rl,
            "wheelspeed_rr": msg.wheelspeed_rr,
            "pdc_dist_front": msg.pdc_dist_front,
            "pdc_dist_rear": msg.pdc_dist_rear,
            "pdc_dist_fls": msg.pdc_dist_fls,
            "pdc_dist_frs": msg.pdc_dist_frs,
            "pdc_dist_rls": msg.pdc_dist_rls,
            "pdc_dist_rrs": msg.pdc_dist_rrs,
        }
        # vehicleinfo_dict=rosidl_runtime_py.message_to_ordereddict(msg)
        vehiclereport_list.append(vehiclereport_dict)
    data_path = os.path.join(output_vehiclereport_topic_path,
                             'vehiclereport.json')
    with refile.smart_open(data_path, "w") as w:
        json.dump(vehiclereport_list, w)
    timestamplist = os.path.join(output_vehiclereport_topic_path,
                                 'timestamplist.txt')
    with refile.smart_open(timestamplist, "w") as w:
        for i in vehiclereport_list:
            w.write(i["timestamp"] + "\n")


def deal_with_ins(topic_name, output_type_folder, topic_id, conn, topic_type):
    output_name = topic_name.replace("/", "#")
    output_ins_topic_path = os.path.join(output_type_folder, output_name)
    count = 10
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)
    ins_list = []
    for row in tqdm(list(cursor)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        #             # 存成二进制

        localization = msg.localization
        pose_dict = {
            "utm_id":
            localization.utm_id,
            "position":
            get_point(localization.position),
            "orientation": {
                "x": localization.orientation.x,
                "y": localization.orientation.y,
                "z": localization.orientation.z,
                "w": localization.orientation.w,
            },
            "linear_velocity":
            get_point(localization.linear_velocity),
            "linear_acceleration":
            get_point(localization.linear_acceleration),
            "angular_velocity":
            get_point(localization.angular_velocity),
            "heading":
            localization.heading,
            "linear_acceleration_vrf":
            get_point(localization.linear_acceleration_vrf),
            'angular_velocity_vrf':
            get_point(localization.angular_velocity_vrf),
            "euler_angles":
            get_point(localization.euler_angles),
        }
        lon, lat, height = msg.lat, msg.lon, msg.height
        if msg.lat == 0:
            t_point = get_point(localization.position)
            utm_wgs_proj = pyproj.Proj(proj="utm",
                                       zone=localization.utm_id,
                                       ellps="WGS84",
                                       preserve_units=True)
            lon, lat = utm_wgs_proj(t_point['x'], t_point['y'], inverse=True)
            height = t_point['z']
        ins_dict = {
            "timestamp": timestamp,
            "ins_status": msg.ins_status,
            "lat": lat,
            "lon": lon,
            "height": height,
            "localization": pose_dict
        }
        ins_list.append(ins_dict)
    data_path = os.path.join(output_ins_topic_path, 'ins.json')
    with refile.smart_open(data_path, "w") as w:
        json.dump(ins_list, w)
    timestamplist = os.path.join(output_ins_topic_path, 'timestamplist.txt')
    with refile.smart_open(timestamplist, "w") as w:
        for i in ins_list:
            w.write(i["timestamp"] + "\n")


def deal_with_gps(topic_name, output_gps_folder, topic_id, conn, topic_type):
    output_name = topic_name.replace("/", "#")
    output_gps_topic_path = os.path.join(output_gps_folder, output_name)
    count = 10
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)

    gps_list = []
    for row in tqdm(list(cursor)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        #             # 存成二进制

        gps_dict = {
            "timestamp": timestamp,
            "solution_status": msg.solution_status,
            "position_type": msg.position_type,
            "lat": msg.lat,
            "lon": msg.lon,
            "height_msl": msg.height_msl,
            "undulation": msg.undulation,
            "datum_id": msg.datum_id,
            "lat_sigma": msg.lat_sigma,
            "lon_sigma": msg.lon_sigma,
            "height_sigma": msg.height_sigma,
            # "base_station_id": msg.base_station_id,
            "diff_age": msg.diff_age,
            "solution_age": msg.solution_age,
            "num_satellites_tracked": msg.num_satellites_tracked,
            "num_satellites_used_in_solution":
            msg.num_satellites_used_in_solution,
            "num_gps_and_glonass_l1_used_in_solution":
            msg.num_gps_and_glonass_l1_used_in_solution,
            "num_gps_and_glonass_l1_and_l2_used_in_solution":
            msg.num_gps_and_glonass_l1_and_l2_used_in_solution,
            "extended_solution_status": msg.extended_solution_status,
            "gps_glonass_used_mask": msg.gps_glonass_used_mask,
            "galileo_beidou_used_mask": msg.galileo_beidou_used_mask,
            "linear_velocity": get_point(msg.linear_velocity),  # 18号
            "velocity_latency": msg.velocity_latency,
        }
        gps_list.append(gps_dict)
    data_path = os.path.join(output_gps_topic_path, 'gps.json')
    with refile.smart_open(data_path, "w") as w:
        json.dump(gps_list, w)
    timestamplist = os.path.join(output_gps_topic_path, 'timestamplist.txt')
    with refile.smart_open(timestamplist, "w") as w:
        for i in gps_list:
            w.write(i["timestamp"] + "\n")


def deal_with_corr_imu(topic_name, output_type_folder, topic_id, conn,
                       topic_type):
    output_name = topic_name.replace("/", "#")
    output_corr_imu_topic_path = os.path.join(output_type_folder, output_name)
    count = 10
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)

    corr_imu_list = []
    for row in tqdm(list(cursor)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        #             # 存成二进制

        imu = msg.imu
        pose_dict = {
            # "utm_id": imu.utm_id,
            # "position": get_point(imu.position),
            # "orientation": {
            #     "x": imu.orientation.x,
            #     "y": imu.orientation.y,
            #     "z": imu.orientation.z,
            #     "w": imu.orientation.w,
            # },
            # "linear_velocity": get_point(imu.linear_velocity),
            # "linear_acceleration": get_point(imu.linear_acceleration),
            # "angular_velocity": get_point(imu.angular_velocity),
            # "heading": imu.heading,
            "linear_acceleration_vrf": get_point(imu.linear_acceleration_vrf),
            'angular_velocity_vrf': get_point(imu.angular_velocity_vrf),
            # "euler_angles": get_point(imu.euler_angles),
        }

        corr_imu_dict = {
            "timestamp": timestamp,
            "imu": pose_dict,
        }
        corr_imu_list.append(corr_imu_dict)
    data_path = os.path.join(output_corr_imu_topic_path, 'corr_imu.json')
    with refile.smart_open(data_path, "w") as w:
        json.dump(corr_imu_list, w)
    timestamplist = os.path.join(output_corr_imu_topic_path,
                                 'timestamplist.txt')
    with refile.smart_open(timestamplist, "w") as w:
        for i in corr_imu_list:
            w.write(i["timestamp"] + "\n")


def deal_with_odom(topic_name, output_type_folder, topic_id, conn, topic_type):
    output_name = topic_name.replace("/", "#")
    output_odom_topic_path = os.path.join(output_type_folder, output_name)
    count = 10
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)

    odom_list = []
    for row in tqdm(list(cursor)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        #             # 存成二进制
        odom_dict = {
            "timestamp": timestamp,
            "pose": {
                "pose": {
                    "position": {
                        "x": msg.pose.pose.position.x,
                        "y": msg.pose.pose.position.y,
                        "z": msg.pose.pose.position.z,
                    },
                    "orientation": {
                        "x": msg.pose.pose.orientation.x,
                        "y": msg.pose.pose.orientation.y,
                        "z": msg.pose.pose.orientation.z,
                        "w": msg.pose.pose.orientation.w,
                    },
                },
            },
            "twist": {
                "twist": {
                    "angular": {
                        "x": msg.twist.twist.angular.x,
                        "y": msg.twist.twist.angular.y,
                        "z": msg.twist.twist.angular.z,
                    },
                    "linear": {
                        "x": msg.twist.twist.linear.x,
                        "y": msg.twist.twist.linear.y,
                        "z": msg.twist.twist.linear.z,
                    },
                }
            }
        }
        odom_list.append(odom_dict)
    data_path = os.path.join(output_odom_topic_path, 'odom.json')
    with refile.smart_open(data_path, "w") as w:
        json.dump(odom_list, w)
    timestamplist = os.path.join(output_odom_topic_path, 'timestamplist.txt')
    with refile.smart_open(timestamplist, "w") as w:
        for i in odom_list:
            w.write(i["timestamp"] + "\n")


def deal_with_lidar(topic_name, output_type_folder, topic_id, conn, topic_type,
                    all_topics_info, bag_path):

    # 点云先要从difop里面获取，雷达标定数据之后解包
    if topic_name == '/rslidar_packets':
        # ###激光理论垂直垂直角度 w
        # vert_angle_list_ = [
        #     -1356, -109, -439, -29, -359, -579, 51, -279, 351, -498, -199, 506, -419,
        #     -1958, -129, -339, -715, -49, -259, -599, 31, -179, -519, -99, -2500, -19,
        #     -765, 61, -269, 141, -189, -1604, -119, -685, -39, 41, -289, 656, 121,
        #     -208, -835, -69, -399, -619, 11, -319, -539, 91, -239, -459, -159, -379,
        #     251, -1034, -89, -299, -9, -219, -559, 71, -139, 1150, -479, -58, -1174,
        #     21, -650, 101, -229, 181, -149, 900, -924, -79, 1, 81, -249, 1500, 161, -169
        # ]
        # ##激光理论水平偏移角度 delta
        # hori_angle_list_ = [
        #     595, 425, 255, 425, 255, 595, 425, 255, 85, 595, 255, 85, 595, 255, 85,
        #     595, 255, 85, 595, 255, 85, 595, 255, 595, 85, 595, 85, 595, 425, 595, 425,
        #     425, 255, 425, 255, 255, 85, 595, 255, 85, -85, -255, -425, -85, -255,
        #     -425, -85, -255, -425, -85, -425, -85, -255, -425, -595, -85, -595, -85,
        #     -425, -595, -85, -255, -425, -85, -595, -85, -595, -85, -255, -85, -255,
        #     -425, -255, -425, -425, -425, -595, -85, -425, -595
        # ]
        output_name = 'middle_lidar'
        lidar_id = 1  # left_lidar :0 right_lidar:2
        nr_beams = 80  # left_lidar :32 right_lidar:32
        rs = RSLidarDecoder('RS80')
        # rs.set_calibration(vert_angle_list_, hori_angle_list_)
        # 通过difop标定,只需调用一次，通常都是一样的
        for _topic_info in all_topics_info:
            _topic_id, _topic_name, _topic_type = _topic_info[:3]
            if _topic_name == '/rslidar_packets_difop':
                sql = "select * from messages where topic_id={}".format(
                    _topic_id)
                cursor = conn.execute(sql)
                for row in cursor:
                    msg = deserialize_message(row[3], get_message(_topic_type))
                    ret = rs.decode_difop(msg.data)

                    if ret:
                        break

    elif topic_name == '/left/rslidar_packets':
        output_name = 'left_lidar'
        lidar_id = 0  # left_lidar :0 right_lidar:2
        nr_beams = 32  # left_lidar :32 right_lidar:32
        rs = RSLidarDecoder('RSBP')

        # 通过difop标定,只需调用一次，通常都是一样的
        for _topic_info in all_topics_info:
            _topic_id, _topic_name, _topic_type = _topic_info[:3]
            if _topic_name == '/left/rslidar_packets_difop':
                sql = "select * from messages where topic_id={}".format(
                    _topic_id)
                cursor = conn.execute(sql)
                for row in tqdm(cursor):
                    msg = deserialize_message(row[3], get_message(_topic_type))
                    ret = rs.decode_difop(msg.data)
                    if ret:
                        break
    elif topic_name == '/right/rslidar_packets':
        output_name = 'right_lidar'
        lidar_id = 2  # left_lidar :0 right_lidar:2
        nr_beams = 32  # left_lidar :32 right_lidar:32
        rs = RSLidarDecoder('RSBP')
        # 通过difop标定,只需调用一次，通常都是一样的
        for _topic_info in all_topics_info:
            _topic_id, _topic_name, _topic_type = _topic_info[:3]
            if _topic_name == '/right/rslidar_packets_difop':
                sql = "select * from messages where topic_id={}".format(
                    _topic_id)
                cursor = conn.execute(sql)
                for row in tqdm(cursor):
                    msg = deserialize_message(row[3], get_message(_topic_type))
                    ret = rs.decode_difop(msg.data)
                    if ret:
                        break
    else:
        assert False, topic_name
    output_lidar_topic_path = os.path.join(output_type_folder, output_name)

    nori_path = os.path.join(bag_path, 'pointcloud.nori')
    pointcloud_norilist = os.path.join(output_lidar_topic_path,
                                       'pointcloud_norilist.txt')

    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)

    cursor_list = list(cursor)
    if not cursor_list:
        return
    result = []

    nw = Nori2RemoteArchiveWriter(file_uri=nori_path,
                                  append_mode=True,
                                  remote_writer_mode=2)
    keys = []
    contents = []
    # nw = nori.remotewriteopen(nori_path, mode=2)
    for idx, row in tqdm(enumerate(cursor_list)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        #             # 存成二进制
        vec = []
        for _msg in msg.packets:
            x = rs.decode_msop(_msg.data)
            vec.append(x)
        points = np.concatenate(vec)

        dt = np.dtype({
            'names': ['x', 'y', 'z', 'i', 'r', 't', 'lidar_id', 'echo_id'],
            'formats': [
                np.float32, np.float32, np.float32, np.uint8, np.uint8,
                np.float64, np.uint8, np.uint8
            ]
        })
        data = np.zeros((points.shape[0], ), dtype=dt)
        data['x'] = points['x']
        data['y'] = points['y']
        data['z'] = points['z']
        data['i'] = points['intensity']
        data['r'] = points['ring']
        data['t'] = points['timestamp']
        data['lidar_id'] = np.ones(
            (points.shape[0]), dtype=np.uint8) * lidar_id

        # points 的维数，应该是动态的
        tmp_points = points.reshape(-1, 2, nr_beams)
        nr_points_per_beam = tmp_points.shape[0]
        first_echo = np.ones(
            (nr_points_per_beam, 1, nr_beams), dtype=np.uint8) * 1
        second_echo = np.ones(
            (nr_points_per_beam, 1, nr_beams), dtype=np.uint8) * 2
        points_echo = np.concatenate((first_echo, second_echo),
                                     axis=1).reshape(-1)  # N

        data['echo_id'] = points_echo
        # valid = (~np.isnan(data['x'])) & (~np.isnan(data['y'])) & (
        #     ~np.isnan(data['z']))
        # data = data[valid]

        f = io.BytesIO()
        np.save(f, data)
        f.seek(0)

        keys.append(timestamp)
        contents.append(f.read())
        # nw.async_put(partial(callback, result, idx, timestamp),
        #              f.read(),
        #              filename=timestamp)
    # nw.join()
    # result = sorted(result, key=lambda x: x[0])
    results = nw.extend_contents(keys=keys, contents=contents)
    nw.close()
    assert len(results) == len(keys) == len(contents)

    with refile.smart_open(pointcloud_norilist, "w") as w:
        # for idx, timestamp, nori_id in result:
        for timestamp, nori_id in zip(keys, results):
            w.write(timestamp + '#' + nori_id + '\n')


def deal_with_radar(topic_name, output_type_folder, topic_id, conn,
                    topic_type):
    output_name = topic_name.replace("/", "#")
    output_radar_topic_path = os.path.join(output_type_folder, output_name)
    count = 10
    sql = "select * from messages where topic_id={}".format(topic_id)
    cursor = conn.execute(sql)

    radar_list = []
    for row in tqdm(list(cursor)):
        msg = deserialize_message(row[3], get_message(topic_type))
        timestamp = str(msg.header.stamp.sec).zfill(10) + '.' + str(
            msg.header.stamp.nanosec).zfill(9)
        radar_objects = msg.objects
        radar_objects_list = []
        for _object in radar_objects:
            position_dict = {
                "pose": {
                    "position": get_point(_object.position.pose.position),
                    "orientation":
                    get_point(_object.position.pose.orientation),
                },
                "covariance": _object.position.covariance.tolist(),
            }
            relative_velocity_dict = {
                "twist": {
                    "angular":
                    get_point(_object.relative_velocity.twist.angular),
                    "linear":
                    get_point(_object.relative_velocity.twist.linear),
                },
                "covariance": _object.relative_velocity.covariance.tolist(),
            }
            relative_acceleration_dict = {
                "accel": {
                    "angular":
                    get_point(_object.relative_acceleration.accel.angular),
                    "linear":
                    get_point(_object.relative_acceleration.accel.linear),
                },
                "covariance":
                _object.relative_acceleration.covariance.tolist(),
            }
            _object_dict = {
                "id": _object.id,
                "position": position_dict,
                "relative_velocity": relative_velocity_dict,
                "relative_acceleration": relative_acceleration_dict,
                "length": _object.length,
                "width": _object.width,
                "orientation_angle": _object.orientation_angle,
                "rcs": _object.rcs,
                "dynamic_property": _object.dynamic_property,
                "class_type": _object.class_type,
                "meas_state": _object.meas_state,
                "prob_of_exist": _object.prob_of_exist,
            }
            radar_objects_list.append(_object_dict)
        radar_list.append({
            "timestamp": timestamp,
            "objects": radar_objects_list,
        })
    data_path = os.path.join(output_radar_topic_path, 'radar.json')
    with refile.smart_open(data_path, "w") as w:
        json.dump(radar_list, w)
    timestamplist = os.path.join(output_radar_topic_path, 'timestamplist.txt')
    with refile.smart_open(timestamplist, "w") as w:
        for i in radar_list:
            w.write(i["timestamp"] + "\n")


def parse_bag(bag_folder, output_folder, bagname):
    if bag_folder[-1] == '/':
        batch_name = os.path.basename(bag_folder[:-1])
    else:
        batch_name = os.path.basename(bag_folder)
    src = os.path.join(bag_folder, bagname)
    bagfile = os.path.join('/tmp', batch_name, bagname)
    # cmd = "aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp {} {}".format(
    #     src, bagfile)
    # p_download = subprocess.run(cmd, shell=True)
    refile.smart_copy(src, bagfile)
    print(bagfile)

    if not refile.smart_exists(bagfile):
        print("download {} failed".formta(bagfile))

    conn = sqlite3.connect(bagfile)
    sql = "select * from topics"
    all_topics_info = conn.execute(sql).fetchall()

    # 尝试两次
    retry = 3
    while retry > 0:
        retry -= 1

        # 遍历sql包，获得topic_id，topic_name,和topic_type的映射
        # topic_info长度是4或5，取决于是否存入额外的meta

        finish_flag = True
        for topic_info in all_topics_info:
            topic_id, topic_name, topic_type = topic_info[:3]
            print(
                topic_id,
                topic_name,
                topic_type,
            )
            # 需要解析的类别
            if topic_type not in topictype2output:
                continue
            output_type_folder = os.path.join(output_folder, batch_name,
                                              bagname,
                                              topictype2output[topic_type])
            bag_path = os.path.join(output_folder, batch_name, bagname)
            print('################################################################')
            print(bag_path)
            # 通过一个*.txt文件，检测是否已经解包好
            if check_s3_nori_num(conn, output_folder, topic_id, topic_name,
                                 topic_type, output_type_folder):
                finish_flag = False
                break
            if topic_type == 'sensor_msgs/msg/CompressedImage':
                deal_with_image(topic_name, output_type_folder, topic_id, conn,
                                topic_type, bag_path)
            elif topic_type == 'devastator_control_msgs/msg/VehicleReportInfo':
                deal_with_vehiclereport(topic_name, output_type_folder,
                                        topic_id, conn, topic_type)
            elif topic_type == 'nav_msgs/msg/Odometry':
                deal_with_odom(topic_name, output_type_folder, topic_id, conn,
                               topic_type)
            elif topic_type == 'devastator_localization_msgs/msg/CorrectedImu':
                deal_with_corr_imu(topic_name, output_type_folder, topic_id,
                                   conn, topic_type)
            elif topic_type == 'devastator_localization_msgs/msg/RawImu':
                deal_with_rawimu(topic_name, output_type_folder, topic_id,
                                 conn, topic_type)
            elif topic_type == 'devastator_localization_msgs/msg/Ins':
                deal_with_ins(topic_name, output_type_folder, topic_id, conn,
                              topic_type)
            elif topic_type == 'rslidar_msg/msg/RslidarScan':
                deal_with_lidar(topic_name, output_type_folder, topic_id, conn,
                                topic_type, all_topics_info, bag_path)
            elif topic_type == 'devastator_localization_msgs/msg/GnssBestPose':
                deal_with_gps(topic_name, output_type_folder, topic_id, conn,
                              topic_type)
            elif topic_type == 'devastator_perception_msgs/msg/RadarObjectArray':
                deal_with_radar(topic_name, output_type_folder, topic_id, conn,
                                topic_type)
        if finish_flag:
            # 如果完成了就退出retry，否则继续循环
            # 对nori进行加速
            image_nori_path = os.path.join(bag_path, 'image.nori')
            pointcloud_nori_path = os.path.join(bag_path, 'pointcloud.nori')
            cmd = "nori speedup {} --on --replica 2".format(bag_path)
            os.system(cmd)
            cmd = "nori speedup {} --on --replica 2".format(
                pointcloud_nori_path)
            os.system(cmd)
            break
        else:
            time.sleep(10)
    os.remove(bagfile)
    return bagname


@task
def parse_ros2bag(bag_folder: str, output_folder: str):
    print("START parse_ros2bag")

    params = []
    for bagname in tqdm(sorted(refile.s3_listdir(bag_folder))):
        if not bagname.endswith('.db3'):
            continue
        params.append((bag_folder, output_folder, bagname))
    print(params[0])

    # # debug模式
    # # futures = [parse_bag(*param) for param in params]
    # # return 1
    # machine_num = min(len(params), 50)
    # os.makedirs(os.path.join(os.getcwd(), 'rrun_log'), exist_ok=True)
    # # The runner template
    # spec = rrun.RunnerSpec()
    # spec.name = 'parse_ros2image'  # For better display in web portal.
    # spec.log_dir = os.path.join(os.getcwd(), 'rrun_log')
    # spec.scheduling_hint.group = 'users'
    # spec.resources.cpu = 1
    # spec.resources.memory_in_mb = 1024 * 30
    # spec.max_wait_time = 3600 * int(1e9)
    #
    # success = True
    #
    # with rrun.RRunExecutor(spec, machine_num, 1) as executor:
    #     futures = [executor.submit(parse_bag, *param) for param in params]
    #
    #     # Here we iterate futures in creation order.
    #     # If you want to iterate in completion order, replace the following line with
    #     # `for future in concurrent.futures.as_completed(futures):`
    #
    #     for idx, future in enumerate(tqdm(futures)):
    #         try:
    #             bagname = future.result()
    #             print('{}: {}, finish one task'.format(idx, bagname))
    #         except Exception as e:
    #             # Catch remote exception
    #             print('failed {}:{}'.format(idx, bagname), e)
    #             success = False
    # if not success:
    #     print('failed {}:{}'.format(idx, bagname))
    #     exit(1)

    if bag_folder[-1] == '/':
        batch_name = os.path.basename(bag_folder[:-1])
    else:
        batch_name = os.path.basename(bag_folder)
    parse_data_path = os.path.join(output_folder, batch_name)

    # 写一行记录原始包储存位置
    with refile.smart_open(
            os.path.join(output_folder, batch_name, 'src_bag.txt'), 'w') as w:
        w.write(bag_folder)
    parse_data_path = parse_data_path + '/'

    user_path = os.path.join(output_folder, batch_name, 'user.txt')
    with refile.smart_open(user_path, 'w') as w:
        w.write(getpass.getuser())

    commit_path = os.path.join(output_folder, batch_name, 'commit_id.txt')
    os.system("git rev-parse HEAD > commit_id.txt")
    os.system(
        "aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp commit_id.txt {}".
        format(commit_path))
    return parse_data_path


@flow(
    project=
    'reform_test',  # 目前只支持项目为 reform/reform_test 会在 tesla 上显示，之后会支持多 project 需求
    name='parse_ros2image')
def run_flow():
    # bag_file = "/data/transformer/washing_machine/scripts/hjb_four_cam_v2.bag/hjb_four_cam_v2.bag_0.db3"
    # output_folder = 'hjb_four_cam_v2.bag_parse'
    parse_ros2bag()


if __name__ == '__main__':
    # 运行
    # run_flow.cli()
    parse_ros2bag(bag_folder='s3://czy1yzc/debug14', output_folder='s3://parse_data14/debug14')
