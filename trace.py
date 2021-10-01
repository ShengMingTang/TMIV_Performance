# airsim version is 1.5.0, use system-wise airsim
# import setup_path
import airsim
import numpy as np
from pathlib import Path
import csv
import cv2
import time
import math
import os
import json
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
    with open(str(outDir/'para.json'), 'w') as f:
        j = {
            'bndbox': bndbox,
            'vel': vel,
            'sampleRate': sampleRate,
            'time': time,
            'wt_max': wt_max,
            'outDir': outDir
        }
        json.dump(j, f)
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
take: True then store pictures
'''
def takePhoto(dir, play=False, take=False):
    client = airsim.VehicleClient()
    client.confirmConnection()
    dir = Path(dir)
    pathcsv = dir/'pose.csv'
    imgDir = dir/'img'
    imgDir.mkdir(exist_ok=True)
    lastT = 0
    hasValidFlag=False
    with open(pathcsv) as f:
        reader = csv.reader(f)
        headers = next(reader)
        hasValidFlag = ('valid' in headers)
        for i, row in enumerate(reader):
            valid = 1
            if hasValidFlag:
                t, x, y, z, roll, pitch, yaw, valid = [float(r) for r in row]
            else:
                t, x, y, z, roll, pitch, yaw = [float(r) for r in row]
            if valid:
                roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw)), True)
                if take:
                    img = client.simGetImage("0", airsim.ImageType.Scene)
                    img = cv2.imdecode(airsim.string_to_uint8_array(img), cv2.IMREAD_UNCHANGED)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(str(imgDir/f'{i}.png'), img)
            if play:
                time.sleep(t - lastT)
                lastT = t

def truncateCovertPng2Yuv(csvPath, inImgDir, outDir, resolution, start, end):
    with open(csvPath) as f:
        outCsv = str(outDir / 'pose.csv')
        inImgDir = Path(inImgDir)
        outDir = Path(outDir)
        outDir.mkdir(exist_ok=True)
        with open(outCsv, 'w', newline='') as fw:
            writer = csv.writer(fw)
            rows = csv.reader(f)
            headers = next(rows)
            writer.writerow(headers)
            for i, row in zip(range(end), rows):
                if i >= start and i < end:
                    inFilename = str(inImgDir/f'{i}.png')
                    outFilename = str(outDir/f'{i}_{resolution[0]}x{resolution[1]}_yuv420p10le.yuv')
                    writer.writerow(row)
                    os.system(
                        f"powershell ffmpeg -i {inFilename} -pix_fmt yuv420p10le {outFilename}"
                    )
                    os.system(
                        f"type {outFilename} >> {str(outDir)}\GT_texture_{resolution[0]}x{resolution[1]}_yuv420p10le.yuv"
                    )
                