#read image sequence from external file
from ALP4 import *
import numpy as np
import time
import pandas as pd

#https://github.com/wavefrontshaping/ALP4lib/blob/master/src/ALP4.py for Python API

DMD = ALP4(version = "4.3")

DMD.Initialize()

#FreeSeq() only frees a sequence allocated through this handle, so right after Initialize() it cannot clear on-board memory still in use; only report it
if DMD.DevInquire(ALP_AVAIL_MEMORY)<55924:
    print("Warning: DMD on-board memory is not empty after Initialize(); this session cannot free it")

seqAllocated = False
#everything after Initialize() is inside try/finally, so the sequence and the DMD are released on any error, not only on Ctrl+C
try:
    #loads image sequence imgSeq generated from imageGen
    DMDseq = np.load("img.npy")
    #determines how many images are in DMDseq using the length of the array
    imgNum = int(DMDseq.size/(DMD.nSizeX*DMD.nSizeY))

    DMD.SeqAlloc(nbImg = imgNum, bitDepth = 1)
    seqAllocated = True
    DMD.SeqPut(imgData = DMDseq)
    #sets how long each image should be displayed, in microseconds
    DMD.SetTiming(pictureTime = 1000000)

    DMD.Run()

    #displays the sequence for time.sleep(x) x number of seconds, can safely end DMD operation with Ctrl-C
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass
finally:
    DMD.Halt()
    if seqAllocated:
        DMD.FreeSeq()
    DMD.Free()