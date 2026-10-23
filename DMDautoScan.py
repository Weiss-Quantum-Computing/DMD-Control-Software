#uploads image sequence one at a time, one image per sequence
from ALP4 import *
import numpy as np
import time
import pandas as pd
from pypylon import pylon

DMD = ALP4(version = "4.3")

DMD.Initialize()

#FreeSeq() only frees a sequence allocated through this handle, so right after Initialize() it cannot clear on-board memory still in use; only report it
if DMD.DevInquire(ALP_AVAIL_MEMORY)<55924:
    print("Warning: DMD on-board memory is not empty after Initialize(); this session cannot free it")

camera = None
seqAllocated = False
#peak camera count for each image, in image order
peaks = []
#everything after Initialize() is inside try/finally, so the sequence, the camera and the DMD are released on any error, not only on Ctrl+C
try:
    camera=pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.ExposureAuto.SetValue("Off")
    #sets exposure time of camera
    expotime = 4000
    camera.ExposureTime.SetValue(expotime)

    DMDseq = np.load("img.npy")
    imgNum = int(DMDseq.size/(DMD.nSizeX*DMD.nSizeY))
    imgList = np.split(DMDseq, imgNum)
    #image count starts from 0, so for scan, the pixel range of the current image is [imgCount*width,(imgCount+1)*width)
    #every image is measured (no early stop: a 1-count noise dip used to end the scan before the beam), and the brightest one is reported
    try:
        for imgCount in range(imgNum):
            #DMD.FreeSeq() undoes allocation so DMD.SeqAlloc() must go inside the loop
            DMD.SeqAlloc(nbImg = 1, bitDepth = 1)
            seqAllocated = True
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
            peaks.append(np.amax(result.Array))
            print(peaks[-1])

            DMD.Halt()
            DMD.FreeSeq()
            seqAllocated = False
    #handles Ctrl+C as a way to end the scan early, the images measured so far are still evaluated
    except KeyboardInterrupt:
        print("Scan stopped after {} images".format(len(peaks)))

    if peaks:
        best = int(np.argmax(peaks))
        #centroid of the brightest image and its two neighbours, with the dimmest measured image as background
        lo = max(best-1,0)
        hi = min(best+2,len(peaks))
        weights = np.asarray(peaks[lo:hi], dtype=float)-min(peaks)
        centroid = np.sum(np.arange(lo,hi)*weights)/np.sum(weights) if np.sum(weights)>0 else float(best)
        print("Beam is at image #{} (peak {} counts), centroid of images #{}-#{} is {:.2f}".format(best, peaks[best], lo, hi-1, centroid))
        #take the image number, add starting frame number from imageGen, and multiply by bar width, that will be the left/upper edge of the bar containing the beam
finally:
    if camera is not None and camera.IsOpen():
        camera.Close()
    DMD.Halt()
    if seqAllocated:
        DMD.FreeSeq()
    DMD.Free()