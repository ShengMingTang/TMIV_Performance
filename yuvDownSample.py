from pathlib import Path
import numpy as np
SCENES = ('ArchVizInterior', 'LightroomInteriorDayLight', 'office', 'RealisticRendering', 'XoioBerlinFlat')
DOWN_FACTOR = 9
FRAMESIZE = 2700*1024 # 2430000/900
for scene in SCENES:
    scnDir = Path('random_mobility_trace')/scene
    outDir = Path('random_mobility_trace_downsampled')/f'{scene}_downsampled'
    outDir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        inFilename = scnDir/f'{scene}_YUVideo'/f'{i}_GT_texture_1280x720_yuv420p10le.yuv'
        outFilename = outDir/f'{i}_downsampled_GT_texture_1280x720_yuv420p10le.yuv'
        inPose = scnDir/f'pose{i}.csv'
        outPose = outDir/f'pose{i}.csv'
        csv = np.loadtxt(open(inPose), dtype=np.float64, delimiter=',', skiprows=1)
        csv = csv[::9, ...]
        foutPose = open(outPose, 'w')
        foutPose.write('t,x,y,z,roll,pitch,yaw\n')
        foutPose = open(outPose, 'a')
        np.savetxt(foutPose, csv, delimiter=',')

        fInV = open(inFilename, 'rb')
        fOutV = open(outFilename, 'wb')
        count = 0
        while count < 900:
            bt = fInV.read(FRAMESIZE)
            if count % 9 == 0:
                fOutV.write(bt)
            count += 1
        fInV.close()
        fOutV.close()

