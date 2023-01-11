import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import json

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict(orient='records')
        return json.JSONEncoder.default(self, obj)

dmdX = 1280
dmdY = 800

def setup():
    #allows user to specify m x n beams
    global cols, rows, beamsize, centerX, centerY, calibDict, dfEntries
    cols = int(input("Enter # of beams (columns): "))
    rows = int(input("Enter # of beams (rows): "))

    #Maximizes beam size by default
    maxsizeX = dmdX//cols
    maxsizeY = dmdY//rows
    bs = min(maxsizeX, maxsizeY)
    beamsize = int(input("Enter desired beam size: ") or str(bs))
    if beamsize%2==0:
        beamsize-=1

    #e.g. centerX=400 places the center 400 pixels from the left edge, centerY=200 places the center 200 pixels from the top edge
    centerX = int(input("Enter desired center (x): ") or str(dmdX//2))
    centerY = int(input("Enter desired center (y): ") or str(dmdY//2))

    #Creates m*n beams initialized to "on"
    calibDict = {}
    for i in range(1,cols*rows+1):
        calibDict["b{}".format(i)] = pd.DataFrame([[255]*beamsize]*beamsize)

    #creates dfEntries to access elements of dataframe as tuple
    dfEntries = []
    for j in range(beamsize):
        for k in range(beamsize):
            dfEntries.append((j,k))

def processCamData(camData):
#takes intensity array from Bryant's program

    intenArr = np.array(camData)
    min = intenArr.min()
    calibArr = min/intenArr
    flatCalibArr = np.ravel(calibArr)
    return flatCalibArr

def calibSetup():
#creates a Gaussian weighted dataframe, w0 dependent on beamsize
    global weightdf, maxInt
    weightdf = pd.DataFrame([[0]*beamsize]*beamsize)
    for p in dfEntries:
        center=(beamsize-1)/2
        x = p[0]
        y = p[1]
        w0 = 30
        weight = math.exp(((x-center)**2+(y-center)**2)/(-2*w0**2))
        weightdf.iat[y,x]=weight

    weightdf /= weightdf.to_numpy().sum()

    #sets total intensity
    fullInten = pd.DataFrame([[255]*beamsize]*beamsize)
    maxInten = weightdf.mul(fullInten)
    maxInt = maxInten.to_numpy().sum()

camArray = [[5,4,3,5],[5,3,4,5],[1,3,4,5],[5,2,4,5]]

def calibrate(camArray):
#finds first iteration of calbirated array
#we want updated camera data each time we iterate

    flatCalibArr = processCamData(camArray)
    global calibDict
    intenTol = float(input("Enter the desired tolerance in intensity: ") or "0.1")
    for i in range(1,cols*rows+1):

        inten = flatCalibArr[i-1]
        inten*=100
        currentdf = calibDict["b{}".format(i)]

        #gives random sample of pixels to be turned off
        n = math.floor((1-inten/100)*beamsize*beamsize)
        randoff = random.sample(dfEntries, n)

        for p in randoff:
            row = p[0]
            col = p[1]
            currentdf.at[row,col] = 0

        totInten = weightdf.mul(currentdf)
        totInt = totInten.to_numpy().sum()
        actualInt = (totInt/maxInt)*100

        while actualInt<inten-intenTol or actualInt>inten+intenTol:
            n = math.floor((abs(inten-actualInt)/100*beamsize*beamsize))
            randoff = random.sample(dfEntries, n)

            for p in randoff:
                row = p[0]
                col = p[1]
                if actualInt>inten:
                    currentdf.at[row,col] = 0
                else:
                    currentdf.at[row,col] = 255

            totInten = weightdf.mul(currentdf)
            totInt = totInten.to_numpy().sum()
            actualInt = (totInt/maxInt)*100
        
    with open('calibSave.json', 'w') as fp:
        json.dump(calibDict, fp, cls=JSONEncoder)

    with open('intenSave.txt', 'w') as filehandle:
        json.dump(camArray, filehandle)

    #saves calibDict and camArray to external files calibSave and intenSave

def iterCal(camArray):

    global calibDict
    data = json.load(open('calibSave.json'))
    calibDict = {
        key: pd.DataFrame(data[key])
        for key in data
    }
    
    with open('intenSave.txt', 'r') as filehandle:
        refInten = json.load(filehandle)
    #accesses external file calibSave and set it equal to global calibDict, access intenSave and set it equal to refInten
    flatCalibArr = np.ravel(np.divide(camArray,refInten))
    refFlat = processCamData(refInten)

    intenTol = float(input("Enter the desired tolerance in intensity: ") or "0.1")
    for i in range(1,cols*rows+1):

        inten = flatCalibArr[i-1]
        inten*=100
        refInt = refFlat[i-1]
        refInt*=100
        currentdf = calibDict["b{}".format(i)]
        currentdf.columns = range(currentdf.columns.size)

        totInten = weightdf.mul(currentdf)
        totInt = totInten.to_numpy().sum()
        calibInt = (totInt/maxInt)*100

        #gives random sample of pixels to be turned off
        n = math.floor((abs(inten-refInt))/100*beamsize*beamsize)
        randoff = random.sample(dfEntries, n)

        for p in randoff:
            row = p[0]
            col = p[1]
            if inten>refInt:
                currentdf.iat[row,col] = 0
            else:
                currentdf.iat[row,col] = 255

        totInten = weightdf.mul(currentdf)
        totInt = totInten.to_numpy().sum()
        actualInt = (totInt/maxInt)*100

        newTarget = calibInt-(inten-refInt)

        while actualInt<newTarget-intenTol or actualInt>newTarget+intenTol:
            n = math.floor((abs(newTarget-actualInt)/100*beamsize*beamsize))
            randoff = random.sample(dfEntries, n)

            for p in randoff:
                row = p[0]
                col = p[1]
                if actualInt>newTarget:
                    currentdf.at[row,col] = 0
                else:
                    currentdf.at[row,col] = 255

            totInten = weightdf.mul(currentdf)
            totInt = totInten.to_numpy().sum()
            actualInt = (totInt/maxInt)*100

        print(actualInt)

    with open('calibSave.json', 'w') as fp:
        json.dump(calibDict, fp, cls=JSONEncoder)

    #save calibDict to external file calibSave



#Wraps active area so that entire dataframe is 1280x800
def wrap(active):
    leftWrapSize = centerX-beamsize*cols//2
    rightWrapSize = dmdX-beamsize*cols-leftWrapSize
    upperWrapSize = centerY-beamsize*rows//2
    lowerWrapSize = dmdY-beamsize*rows-upperWrapSize

    leftdf = pd.DataFrame([[0]*leftWrapSize]*(beamsize*rows))
    rightdf = pd.DataFrame([[0]*rightWrapSize]*(beamsize*rows))
    image = pd.concat([leftdf,active,rightdf], axis=1, ignore_index=True)

    upperdf = pd.DataFrame([[0]*dmdX]*upperWrapSize)
    lowerdf = pd.DataFrame([[0]*dmdX]*lowerWrapSize)
    image = pd.concat([upperdf,image,lowerdf], ignore_index=True)

    return image

imgSeq = {}
imgNum = 0

def createLibrary():
    global imgSeq, imgNum
    imgSeq = {}
    imgNum = 0
    alloff()
    allon()
    onebeam()
    checkerone()
    checkertwo()
    return imgSeq

#Combines all beams into one big dataframe called "active"
#beamDict2 is a dictionary of rows of beams
#Creates "all off" image
def alloff():
    global imgSeq, imgNum
    beamDict2 = {}
    imgNum += 1
    for i in range(0,rows):
        arr = []
        for j in range(i*cols+1,(i+1)*cols+1):
            arr.append(pd.DataFrame([[0]*beamsize]*beamsize))
        beamDict2["row{}".format(i+1)]=pd.concat(arr, axis=1, ignore_index=True)

    arr=[]
    for df in beamDict2.values():
        arr.append(df)
    active = pd.concat(arr, ignore_index=True)
    img = wrap(active)
    imgSeq["img{}".format(imgNum)] = img

#Creates "all on" image
def allon():
    global imgSeq, imgNum
    beamDict2 = {}
    imgNum += 1
    for i in range(0,rows):
        arr = []
        for j in range(i*cols+1,(i+1)*cols+1):
            arr.append(calibDict["b{}".format(j)])
        beamDict2["row{}".format(i+1)]=pd.concat(arr, axis=1, ignore_index=True)

    arr=[]
    for df in beamDict2.values():
        arr.append(df)
    active = pd.concat(arr, ignore_index=True)
    img = wrap(active)
    imgSeq["img{}".format(imgNum)] = img

#generates single beam images
def onebeam():
    global imgSeq, imgNum
    for i in range(1, cols*rows+1):
        beamDict = {}
        imgNum += 1
        for j in range(1,cols*rows+1):
            if j==i:
                beamDict["b{}".format(j)] = calibDict["b{}".format(j)]
            else:
                beamDict["b{}".format(j)] = pd.DataFrame([[0]*beamsize]*beamsize)

        beamDict2 = {}
        for k in range(0,rows):
            arr = []
            for l in range(k*cols+1,(k+1)*cols+1):
                arr.append(beamDict["b{}".format(l)])
            beamDict2["row{}".format(k+1)]=pd.concat(arr, axis=1, ignore_index=True)

        arr=[]
        for df in beamDict2.values():
            arr.append(df)
        active = pd.concat(arr, ignore_index=True)
        img = wrap(active)
        imgSeq["img{}".format(imgNum)] = img

#generates first checkerboard pattern
def checkerone():
    global imgSeq, imgNum
    beamDict2 = {}
    imgNum += 1
    for i in range(0,rows):
        arr = []
        for j in range(i*cols+1,(i+1)*cols+1):
            if cols%2==1:
                if j%2==0:
                    arr.append(calibDict["b{}".format(j)])
                else:
                    arr.append(pd.DataFrame([[0]*beamsize]*beamsize))
            else:
                if i%2==0:
                    if j%2==0:
                        arr.append(calibDict["b{}".format(j)])
                    else:
                        arr.append(pd.DataFrame([[0]*beamsize]*beamsize)) 
                else:
                    if j%2==1:
                        arr.append(calibDict["b{}".format(j)])
                    else:
                        arr.append(pd.DataFrame([[0]*beamsize]*beamsize))  

        beamDict2["row{}".format(i+1)]=pd.concat(arr, axis=1, ignore_index=True)

    arr=[]
    for df in beamDict2.values():
        arr.append(df)
    active = pd.concat(arr, ignore_index=True)
    img = wrap(active)
    imgSeq["img{}".format(imgNum)] = img

#generates second checkerboard pattern
def checkertwo():
    global imgSeq, imgNum
    beamDict2 = {}
    imgNum += 1
    for i in range(0,rows):
        arr = []
        for j in range(i*cols+1,(i+1)*cols+1):
            if cols%2==1:
                if j%2==1:
                    arr.append(calibDict["b{}".format(j)])
                else:
                    arr.append(pd.DataFrame([[0]*beamsize]*beamsize))
            else:
                if i%2==1:
                    if j%2==0:
                        arr.append(calibDict["b{}".format(j)])
                    else:
                        arr.append(pd.DataFrame([[0]*beamsize]*beamsize)) 
                else:
                    if j%2==1:
                        arr.append(calibDict["b{}".format(j)])
                    else:
                        arr.append(pd.DataFrame([[0]*beamsize]*beamsize))  

        beamDict2["row{}".format(i+1)]=pd.concat(arr, axis=1, ignore_index=True)

    arr=[]
    for df in beamDict2.values():
        arr.append(df)
    active = pd.concat(arr, ignore_index=True)
    img = wrap(active)
    imgSeq["img{}".format(imgNum)] = img

setup()
calibSetup()
#camArray = getCamData(), each new camArray is retrieved from camera intensity program
camArray = [[5,4,3,5],[5,3,4,5],[1,3,4,5],[5,2,4,5]]
calibrate(camArray)
camArray = [[1.1,0.9,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
iterCal(camArray)


imgs = createLibrary()
image = imgs["img2"]
ig = image.to_numpy()
vis = plt.imshow(ig)
vis.set_cmap('binary')
plt.show()

#converts image sequence into 1D array for input into DMD
def dmdIn():
    DMDseq = []
    for df in createLibrary().values():
        ndArr = df.to_numpy()
        image = np.ravel(ndArr)
        DMDseq.extend(image)
    return DMDseq