import numpy as np
import cv2
from pathlib import Path
import airsim
import msgpackrpc
import sys
import math
import csv
import os


np.set_printoptions(threshold=sys.maxsize)
RESOLUTION = [1280, 720]
SCENE = 'XoioBerlinFlat'
RAND_PATH = Path.home()/'TMIV_Performance'/'random_mobility_trace_revise'/SCENE

class Camera_pose:
    def __init__(self):
        self.name = ""
        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]
def import_cameras_pose(csvfile_PATH):
    cameras_pose = []
    with open(csvfile_PATH, 'r') as csv_f:
        rows = csv.reader(csv_f)
        next(rows)
        for row in rows:
            camera_pose = Camera_pose()
            # camera_pose.name = row[0]
            # camera_pose.position = [float(x) for x in row[1:4]]
            # camera_pose.rotation = [float(x) for x in row[4:7]]
            # camera_pose.name = row[0]
            camera_pose.position = [float(x) for x in row[:3]]
            camera_pose.rotation = [float(x) for x in row[3:]]
            cameras_pose.append(camera_pose)
    return cameras_pose
# def set_camera_pose(client, camera_pose):
#     client.simSetCameraPose("front_center", airsim.Pose(
#         airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)))
#     client.simSetVehiclePose(
#         airsim.Pose(
#             airsim.Vector3r(
#                 camera_pose.position[0], -camera_pose.position[1], -camera_pose.position[2]),
#             airsim.to_quaternion(
#                 camera_pose.rotation[1]*math.pi/180, -camera_pose.rotation[2]*math.pi/180, -camera_pose.rotation[0]*math.pi/180),
#         ),
#         True
#     )

# 0902 version
def set_camera_pose(client, camera_pose):
    client.simSetCameraPose("front_center", airsim.Pose(
    airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)))
    client.simSetVehiclePose(
    airsim.Pose(
    airsim.Vector3r(
    camera_pose.position[0], -camera_pose.position[1], -camera_pose.position[2]),
    airsim.to_quaternion(
    -camera_pose.rotation[1]*math.pi/180, camera_pose.rotation[2]*math.pi/180, -camera_pose.rotation[0]*math.pi/180),
    ),
    True
    )
def GT_main(cameras_pose, num_frames, datai):
    print(msgpackrpc.__version__)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    # os.system(f"powershell mkdir test_miv/GT_tmp")
    IMG_DIR = RAND_PATH/f'img{datai}'
    IMG_DIR.mkdir(exist_ok=True)

    for f_idx, camera_pose in enumerate(cameras_pose):
        set_camera_pose(client, camera_pose)
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])  # scene vision image in uncompressed RGB array
        
        # filename = f"test_miv/GT_tmp/{f_idx}"
        filename = str(IMG_DIR/f'{f_idx}')
        for response in responses:
            print("Type %d, size %d" %
                  (response.image_type, len(response.image_data_uint8)))
            img1d = np.fromstring(
                response.image_data_uint8, dtype=np.uint8)  # get numpy array
            # reshape array to 3 channel image array H X W X 3
            img_rgb = img1d.reshape(response.height, response.width, 3)
            cv2.imwrite(filename + '.png', img_rgb)  # write to png
            # convert RGB video into yuv420p10le
            if(response.image_type == 0):
                os.system(
                    f"powershell ffmpeg -i {filename}.png -pix_fmt yuv420p10le {filename}_{RESOLUTION[0]}x{RESOLUTION[1]}_yuv420p10le.yuv")
                os.system(f"powershell rm {filename}.png")
        # os.system(
        #     f"type test_miv\GT_tmp\{f_idx}.yuv >> test_miv\GT\GT_texture_{RESOLUTION[0]}x{RESOLUTION[1]}_yuv420p10le.yuv")
        os.system(
            f"type {filename}_{RESOLUTION[0]}x{RESOLUTION[1]}_yuv420p10le.yuv >> {str(IMG_DIR)}\GT_texture_{RESOLUTION[0]}x{RESOLUTION[1]}_yuv420p10le.yuv")
        os.system(f"powershell rm {filename}_{RESOLUTION[0]}x{RESOLUTION[1]}_yuv420p10le.yuv")
    # os.system("powershell rm -r test_miv/GT_tmp")

for i in range(10):
    cameras_pose = import_cameras_pose(str(RAND_PATH/f'pose{i}.csv'))
    num_frames = len(cameras_pose)
    GT_main(cameras_pose, num_frames, i)
    # results will be stored at GT
    # os.system("powershell python   recorder_for_MIV.py clean")

# os.system(
    # f"powershell python recorder_for_MIV.py GT {PATH_TO_TRACE}")


        
