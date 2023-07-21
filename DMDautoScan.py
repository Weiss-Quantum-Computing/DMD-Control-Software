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
expotime = 4000
camera.ExposureTime.SetValue(expotime)

DMDseq = np.load("img.npy")
imgNum = int(DMDseq.size/(DMD.nSizeX*DMD.nSizeY))
imgList = np.split(DMDseq, imgNum)
imgCount = -1
brightest = 0 
prev = -1
#image count starts from 0, so for scan, the pixel range of the current image is [imgCount*width,(imgCount+1)*width)
try:
    while brightest>=prev-1:
        imgCount+=1
        print(imgCount)
        #DMD.FreeSeq() undoes allocation so DMD.SeqAlloc() must go inside the loop
        DMD.SeqAlloc(nbImg = 1, bitDepth = 1)
        DMD.SeqPut(imgData = imgList[imgCount])
        DMD.SetTiming(pictureTime = 1000000)
        print("Current image # is " + str(imgCount))
        #takes into account the time between the trigger and the actual projection
        trigToProjDelay = ALP4.SeqInquire(self=DMD, inquireType=ALP_MAX_TRIGGER_IN_DELAY)/1000000
        DMD.Run()
        #ensures that the desired image appears on the camera before taking a picture
        time.sleep(trigToProjDelay)
        #takes a picture using camera, wait for grabResult for a duration depending on the exposure time (with a 1ms buffer)
        result = camera.GrabOne(expotime+1000)
        prev = brightest
        brightest = np.amax(result.Array)
        print(brightest)

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