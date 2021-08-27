import setup_path
import airsim
import numpy as np
from pathlib import Path
import csv
import cv2
import matplotlib.pyplot as plt
import time
import math

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

'''
Coordinate Transform:
Unreal to AirSim: x, y, z = y/100, x,/100 -z/100
roll, pitch, yaw left unchanged
'''

'''
https://hackmd.io/lc8Zo3oeRFqF9VtIIZmrew?view
Distances are in meter, angles are in degree
bndbox: [[minX, maxX], ...] (6,2) as shown in hackMD
vel: [[minTrans, maxTrans], [minRot, maxRot]]
sampleRate:
time: total record time
wt_max: maximum wait time in frames
outDir: output directory
'''
def genPose(bndbox, vel, sampleRate, time, wt_max, outDir):
    def randomWayPoint(bndbox, vel, sampleRate, wt_max):
        pose = np.zeros((6,))
        cTrans, cRot = 0, 0
        while True:
            if cTrans <= 0:
                wpTrans = np.random.uniform(size=(3,))
                wpTrans = wpTrans*bndbox[:3, 0] + (1-wpTrans)*bndbox[:3, 1]
                vTrans = np.random.uniform()
                vTrans = vTrans*vel[0, 0] + (1-vTrans)*vel[0, 1]
                cTrans = int(np.linalg.norm( wpTrans - pose[:3] ) / vTrans)
                cTrans += np.random.randint(wt_max)
                vTrans = (wpTrans - pose[:3]) * vTrans / (np.linalg.norm(wpTrans - pose[:3])+1e-6)
                # print(f'wpTrans={wpTrans}')
            if cRot <= 0:
                wpRot = np.random.uniform(size=(3,))
                wpRot = wpRot*bndbox[3:, 0] + (1-wpRot)*bndbox[3:, 1]
                vRot = np.random.uniform()
                vRot = vRot*vel[1, 0] + (1-vRot)*vel[1, 1]
                cRot = int(np.linalg.norm( wpRot - pose[3:] ) / vRot)
                cRot += np.random.randint(wt_max)
                vRot = (wpRot - pose[3:]) * vRot / (np.linalg.norm((wpRot - pose[3:]))+1e-6)
                # print(f'wpRot={wpRot}')
            yield pose
            if cTrans != 0:
                pose[:3] += vTrans * (1/sampleRate)
            if cRot:
                pose[3:] += vRot * (1/sampleRate)
            # print(wpTrans, wpRot)
            # print(cTrans, cRot)
            # print(vTrans, vRot)
            cTrans -= 1
            cRot -= 1
    outDir = Path(outDir)
    bndbox = np.array(bndbox)
    vel = np.array(vel)
    gen = randomWayPoint(bndbox, vel, sampleRate, wt_max)
    data = []
    if outDir.exists() is False:
        outDir.mkdir()
    with open(str(outDir/'pose.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        for i, pose in zip(range(sampleRate*time), gen):
            x, y, z, roll, pitch, yaw = pose 
            writer.writerow([i*(1/sampleRate), x, y, z, roll, pitch, yaw])
            data.append([x, y, z, roll, pitch, yaw])
        return data
'''
dir: input directory in generate()
play: True then pause between frames
isFromHMDCord: True if coordinate system is captured from Unreal Engine directly
'''
def takePhoto(dir, play=False, isFromHMDCord=False, take=False):
    client = airsim.VehicleClient()
    client.confirmConnection()
    dir = Path(dir)
    pathcsv = dir/'pose.csv'
    imgDir = dir/'img'
    imgDir.mkdir(exist_ok=True)
    lastT = 0
    with open(pathcsv) as f:
        reader = csv.reader(f)
        headers = next(reader)
        for i, row in enumerate(reader):
            t, x, y, z, roll, pitch, yaw = [float(r) for r in row]
            roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
            if isFromHMDCord:
                x, y, z = x, y, -z
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw)), True)
            if take:
                img = client.simGetImage("0", airsim.ImageType.Scene)
                img = cv2.imdecode(airsim.string_to_uint8_array(img), cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(str(imgDir/f'{i}.png'), img)
            if play:
                time.sleep(t - lastT)
                lastT = t

if __name__ == '__main__':
    # Plot
    # ==============================================================================================================================
    takePhoto(dir=Path.home()/'TMIV_Performance'/'HMD_TEST_Random', play=False, isFromHMDCord=True, take=False)
    
    # Generate
    # ==============================================================================================================================
    # OUT_DIR = Path.home()/'TMIV_Performance'/'data'
    # BND_BOX = [
    #     [-3, 3], # min/max X
    #     [-3, 3], # min/max Y
    #     [-5,-3], # min/max Z
    #     [0,0], # min/max Pitch
    #     [0,0], # min/max Yaw
    #     [0, 60], # min/max Roll
    # ]
    # VEL = [
    #     [1e-4, 1e-3], # minVelTrans, maxVelTrans
    #     [1, 30], # minVelRot, maxVelRot
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
    
    # Position Only
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
