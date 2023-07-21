import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools
import beamtracking

#uses 2D numpy arrays instead of dataframes, calibrated stored as list 49 separate calibrated beams

dmdX = 1280
dmdY = 800
dmdcenX = dmdX//2
dmdcenY = dmdY//2

def setup():
    #allows user to specify m x n beams
    global cols, rows, beamsize, centerX, centerY, calibrated, minX, minY     
    cols = 7
    rows = 7

    #enter desired square beamsize, before transformation into parallelograms
    beamsize = 75
    if beamsize%2==0:
        beamsize-=1

    #e.g. centerX=400 places the center 400 pixels from the left edge, centerY=200 places the center 200 pixels from the top edge
    centerX = 640
    centerY = 400

    minX = centerX-cols*beamsize//2
    minY = centerY-rows*beamsize//2

    #Creates "calibrated" which is our baseline array which will be our reference for the calibrated "on" beams
    #will be all 0 outside of beamcalib area, during the setup it will set all beams to full and calibrate it down from there
    calibrated = []

def processCamData(camData):
#takes intensity array from camera program and flattens and normalizes it, assuming camData is m by n array of unnormalized intensities

    intenArr = np.array(camData)
    min = intenArr.min()
    calibArr = min/intenArr
    return calibArr

def calibSetup():
#creates a Gaussian weighted dataframe, w0 eventually obtained from beamtracking
#we want to turn a number of pixels off to equalize intensities between beams, but each pixel is weighted differently since the beam is Gaussian
#this just gives a ballpark estimate of which pixels should be turned off, but in the end we need to resample the camera and iterate
    global parGauss, calibrated
    center = (beamsize-1)/2
    w0 = 30 
    weightArr = [[math.exp(((x-center)**2+(y-center)**2)/(-2*w0**2)) for x in np.arange(beamsize)] for y in np.arange(beamsize)]
    np.asarray(weightArr)
    #parGauss will be a full image with all 49 beams on, with a Gaussian weight projected over each individual parallelogram beam
    parGauss = np.zeros((dmdY,dmdX))

    #for each square beam, transforms their positions as vectors using the COB matrix so that the beams become parallelograms
    for i in range(rows):
        for j in range(cols):
                beamcalib = np.full((dmdY,dmdX), False)
                angle = 0.77
                #since out-of-plane rotation occurs about the axis going through the DMD center with pixel slope of 1, we shift the origin of pixel position vectors to the center
                active = np.zeros((dmdY,dmdX))
                active[minY+i*beamsize:minY+(i+1)*beamsize,minX+j*beamsize:minX+(j+1)*beamsize] = weightArr
                v = np.where(active!=0)
                veclist = np.asarray(v)
                #shifting the origin for the position vectors before the transformation
                veclist[0]=np.subtract(veclist[0],400)
                veclist[1]=np.subtract(veclist[1],640)
                COB = [[(1/math.cos(angle)-1)/2, (1/math.cos(angle)+1)/2], [(1/math.cos(angle)+1)/2, (1/math.cos(angle)-1)/2]]
                newpos = (np.around(np.matmul(COB, veclist))).astype(int)

                newpos[0]=np.add(newpos[0],400)
                newpos[1]=np.add(newpos[1],640)
                veclist[0]=np.add(veclist[0],400)
                veclist[1]=np.add(veclist[1],640)

                for vec,pos in zip(veclist.T,newpos.T):
                    #assigns each pixel in the parallelogram the weight of its corresponding pixel in a square Gaussian, remember that each pixel in the square each mapped to multiple pixels in the parallelogram
                    #notice that without adding some redundancy, there are diagonal gaps due to Python rounding in the final parallelogram because the area gets scaled up, so one-to-one mapping will not cover the full parallelogram
                    #so we also fill in each pixel above and to the right of all the new vectors calculated from the one-to-one mapping
                    parGauss[pos[0],pos[1]]=active[vec[0],vec[1]]
                    parGauss[pos[0]-1,pos[1]]=active[vec[0],vec[1]]
                    parGauss[pos[0],pos[1]+1]=active[vec[0],vec[1]]
                    parGauss[pos[0]-1,pos[1]+1]=active[vec[0],vec[1]]
                    #while looping through all the transformed positions, we also populate the calibrated array with full intensity single beam images
                    beamcalib[pos[0],pos[1]]=True
                    beamcalib[pos[0]-1,pos[1]]=True
                    beamcalib[pos[0],pos[1]+1]=True
                    beamcalib[pos[0]-1,pos[1]+1]=True
            
                active[minY+i*beamsize:minY+(i+1)*beamsize,minX+j*beamsize:minX+(j+1)*beamsize] = np.zeros((beamsize,beamsize))
                calibrated.append(beamcalib)

    #normalizes Gaussian
    parGauss /= np.sum(parGauss)
    parGauss *= rows*cols


camArray = [[5,2,3,5],[4,3,4,2],[1,3,2,5],[1,2,4,3]]

#calibrates by taking away random pixels one by one and counting up their intensities according to Gaussian weighting
def calibrate(camArray):
#finds first iteration of calibrated array, which is the equalized baseline pattern that we will use for all images
#we want updated camera data each time we iterate

    calibArr = processCamData(camArray)
    global calibrated
    #gramlist stores list of vectors (with origin shifted to corner of beamcalib area) in each parallelogram
    gramlist = []

    for i,row in enumerate(calibArr):
        for j,inten in enumerate(row):
                beamindex = i*cols+j
                currentbeam = calibrated[beamindex]

                angle = 0.77
                active = np.zeros((dmdY,dmdX))
                active[minY+i*beamsize:minY+(i+1)*beamsize,minX+j*beamsize:minX+(j+1)*beamsize] = np.full((beamsize,beamsize), True)
                v = np.where(active)
                veclist = np.asarray(v)

                veclist[0]=np.subtract(veclist[0],400)
                veclist[1]=np.subtract(veclist[1],640)

                COB = [[(1-1/math.cos(angle))/2, (1/math.cos(angle)+1)/2], [(1/math.cos(angle)+1)/2, (1-1/math.cos(angle))/2]]
                newpos = (np.around(np.matmul(COB, veclist))).astype(int)
                newpos[0]=np.add(newpos[0],400)
                newpos[1]=np.add(newpos[1],640)
                active[i*beamsize:(i+1)*beamsize,j*beamsize:(j+1)*beamsize] = np.full((beamsize,beamsize), False)

                #poslist is just gramlist with duplicates
                poslist = []
                for pos in newpos.T:
                    poslist.append([pos[0],pos[1]])
                    poslist.append([pos[0]-1,pos[1]])
                    poslist.append([pos[0],pos[1]+1])
                    poslist.append([pos[0]-1,pos[1]+1])
                
                #gets rid of duplicates from poslist
                poslist.sort()
                dupeless = list(poslist for poslist,_ in itertools.groupby(poslist))
                gramlist.append(dupeless)

                intencount = 0
                while intencount<1-inten:
                    pos = random.choice(dupeless)
                    if currentbeam[pos[0],pos[1]]:
                        currentbeam[pos[0],pos[1]]=False
                        intencount+=parGauss[pos[0],pos[1]]
                
    np.save("calibSave.npy", calibrated)
    np.save("intenSave.npy", camArray)
    np.save('gramSave.npy', np.array(gramlist,dtype=object), allow_pickle=True)
    #saves calibrated,camArray, and gramlist to external files calibSave,intenSave, and gramSave so that we can access them between instances of accessing camera

def iterCal(camArray):

    global calibrated
    calibrated = np.load("calibSave.npy")
    #refInten is just the camArray from the first calibration
    refInten = np.load("intenSave.npy")
    gramlist = np.load("gramSave.npy", allow_pickle=True)
    #accesses external file calibSave and set it equal to global calibrated, access intenSave and set it equal to refInten, access gramSave and set it equal to gramlist
    calibArr = np.divide(camArray,refInten)
    ref = processCamData(refInten)

    #loops through rows and columns of the two rowsxcols 2D arrays calibArr, ref in parallel
    for i,row in enumerate(zip(calibArr,ref)):
        for j,col in enumerate(zip(row[0],row[1])):
            beamindex = i*cols+j
            currentbeam = calibrated[beamindex]
            intencount = 0
            inten = col[0]
            refInt = col[1]
            dupeless = gramlist[beamindex]

            #simply takes away or adds back the intensity equal to the difference between the desired percentage of original intensity and the actual percentage of original intensity
            #"desired percentage of original intensity" means the percentage needed from the first calibration to make all beams equal
            if refInt<inten:
                while intencount<inten-refInt:
                    pos = random.choice(dupeless)
                    if currentbeam[pos[0],pos[1]]:
                        currentbeam[pos[0],pos[1]]=False
                        intencount+=parGauss[pos[0],pos[1]]                

            if refInt>inten:
                while intencount<refInt-inten:
                    pos = random.choice(dupeless)
                    if not currentbeam[pos[0],pos[1]]:
                        currentbeam[pos[0],pos[1]]=True
                        intencount+=parGauss[pos[0],pos[1]]

    vis = plt.imshow(calibrated[2])
    vis.set_cmap('binary')
    plt.show()

    np.save("calibSave.npy", calibrated)

    #save calibDict to external file calibSave


#######CODE HERE#######

setup()
calibSetup()
camArray = np.ones((7,7))
calibrate(camArray)

#example of iterated calibration, remember to change cols and rows to 4x4 in setup()
""" camArray = [[5,2,2,5],[4,3,4,2],[1,3,2,5],[1,2,4,3]]
calibrate(camArray)
camArray = [[1,1.5,0.5,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
iterCal(camArray) """

#example of taking real-time camera input
""" camArray = beamtracking.getIntenList()
calibrate(camArray) """


#######CODE HERE#######