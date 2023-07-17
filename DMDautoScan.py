#uploads image sequence one at a time, one image per sequence
from ALP4 import *
import numpy as np
import time
import pandas as pd
from pypylon import pylon

DMD = ALP4(version = "4.3")

DMD.Initialize()

if DMD.DevInquire(ALP_AVAIL_MEMORY)<55924:
    DMD.FreeSeq()

camera=pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureAuto.SetValue("Off")
#sets exposure time of camera
camera.ExposureTime.SetValue(4000)

DMDseq = np.load("img.npy")
imgNum = int(DMDseq.size/(DMD.nSizeX*DMD.nSizeY))
imgList = np.split(DMDseq, imgNum)
imgCount = -1
brightest = 0 
prev = -1
DMD.SeqAlloc(nbImg = 1, bitDepth = 1)
#image count starts from 0, so for scan, the pixel range of the current image is [imgCount*width,(imgCount+1)*width)
try:
    while brightest>=prev:
        imgCount+=1
        DMD.SeqPut(imgData = imgList[imgCount])
        DMD.SetTiming(pictureTime = 1000000)
        print("Current image # is " + str(imgCount))
        DMD.Run()
        #takes a picture using camera
        result = camera.GrabOne()
        prev = brightest
        brightest = np.sum(result.Array)

        DMD.Halt()
        DMD.FreeSeq()
#handles Ctrl+C as a way to end DMD operation safely      
except KeyboardInterrupt:
    pass

camera.Close()
print("Beam is at image #{}".format(imgCount))
#take the image number, add starting frame number from imageGen, and multiply by bar width, that will be the left/upper edge of the bar containing the beam

DMD.Halt()
if DMD.DevInquire(ALP_AVAIL_MEMORY)<55924:
    DMD.FreeSeq()
DMD.Free()