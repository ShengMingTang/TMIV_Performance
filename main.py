from trace import * 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
'''
Settings.json

{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "ComputerVision",
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 256,
        "Height": 144,
        "FOV_Degrees": 90
      }
    ]
  }
}

'''

# Batch
# ==============================================================================================================================
# TIME = 30
# WT_MAX = 30

# for i in range(10):
#   OUT_DIR = Path.home()/'TMIV_Performance'/'random_mobility_Trans0105_Rot_5_80'/f'data{i}'
#   BND_BOX = [
#       [0, 0.35], # min/max X
#       [-0.5, 0.5], # min/max Y
#       [-0.5, 0.5], # min/max Z
#       [-80, 80], # min/max Pitch
#       [-80, 80], # min/max Yaw
#       [-80, 80], # min/max Roll
#   ]
#   VEL = [
#       [0.1, 0.5], # minVelTrans, maxVelTrans
#       [5, 80], # minVelRot, maxVelRot
#   ]

#   SAMPLE_RATE = 30 # samepls/sec
#   data = genPose(
#       bndbox= BND_BOX, vel=VEL,
#       sampleRate=SAMPLE_RATE, time=TIME, wt_max=WT_MAX, outDir=OUT_DIR
#   )
# Generate
# ==============================================================================================================================
# OUT_DIR = Path.home()/'TMIV_Performance'/'data'
# BND_BOX = [
#     [0, 0.35], # min/max X
#     [-0.5, 0.5], # min/max Y
#     [-0.5, 0.5], # min/max Z
#     [-80, 80], # min/max Pitch
#     [-80, 80], # min/max Yaw
#     [-80, 80], # min/max Roll
# ]
# VEL = [
#     [0.1, 0.5], # minVelTrans, maxVelTrans
#     [5, 80], # minVelRot, maxVelRot
# ]
# TIME = 5
# WT_MAX = 2

# SAMPLE_RATE = 30 # samepls/sec
# data = genPose(
#     bndbox= BND_BOX, vel=VEL,
#     sampleRate=SAMPLE_RATE, time=TIME, wt_max=WT_MAX, outDir=OUT_DIR
# )

# Plot
# ==============================================================================================================================
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # Position Only
# data = np.array(data)
# ax.plot(data[:, 0], data[:, 1], data[:, 2])
# plt.show()
# plt.clf()
# fig = plt.figure()

# Vector
# https://stackoverflow.com/questions/1568568/how-to-convert-euler-angles-to-directional-vector
# data = np.array(data)
# ax = fig.add_subplot(projection='3d')
# x, y, z = data[:, 0], data[:, 1], data[:, 2]
# data[:, 3:] *= (math.pi/180)
# u = np.cos(data[:, 5]) * np.cos(data[:, 4])
# v = np.sin(data[:, 5]) * np.cos(data[:, 4])
# w = np.sin(data[:, 4])
# ax.quiver(x, y, z, u, v, w, length=0.1)
# plt.show()

# Run
# ==============================================================================================================================
# takePhoto(dir=Path.home()/'TMIV_Performance'/'YuanJun', play=False, take=True)

# Collect Ground Truth from VR captured
root = Path.home()/'TMIV_Performance'/'YuanJun'
RESOLUTION = [1280, 720]
truncateCovertPng2Yuv(root/'pose.csv', root/'img', root/'outImg', RESOLUTION, 82, 82+90)
