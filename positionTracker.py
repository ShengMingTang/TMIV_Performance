import setup_path
import airsim
import numpy as np
from pathlib import Path
import csv
import cv2
import matplotlib.pyplot as plt
import time
import math
client = airsim.VehicleClient()
client.confirmConnection()
while True:
    pose = client.simGetVehiclePose()
    print(pose)
    time.sleep(1)