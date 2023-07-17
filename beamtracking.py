from PIL import Image
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
from numpy import *
from pandas import DataFrame
from scipy.optimize import curve_fit
from pypylon import pylon

def setup():
    global xdim,ydim,sep,df,imgarr

    """ #accessing camera example
    camera=pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.ExposureAuto.SetValue("Off")
    #sets exposure time of camera
    camera.ExposureTime.SetValue(4000)

    result = camera.GrabOne()
    img=result.Array #check datatype?
    imgarr = np.asarray(img)
    camera.Close() """


    image = Image.open(r"C:\Users\rdbb7\DMD Code\hello\ResearchWeiss\test.png")

    imgarr = asarray(image)
    xdim = imgarr.shape[1]
    ydim = imgarr.shape[0]
    sep = 30 #beam separation is expected distance between adjacent centers if entire lattice is filled in
    #for now enter 30 for beam sep
    df = tp.locate(imgarr, 25) #gives locations in form row,col i.e y,x with origin in top left, second argument is just expected odd diameter of beams

#tp.locate() does not return integer positions of peaks
#gets integer positions of peaks from trackpy (checks the four closest pixels from the actual image and selects the brightest), puts them in a tuple list in form y,x
def getcenters():
    centerlist = list()
    for index,row in df.iterrows():
        fy = int(floor(row['y']))
        cy = int(ceil(row['y']))
        fx = int(floor(row['x']))
        cx = int(ceil(row['x']))
        m = max(imgarr[fy][fx],imgarr[fy][cx],imgarr[cy][fx],imgarr[cy][cx])
        if imgarr[fy][fx]==m:
            truey=fy
            truex=fx
        elif imgarr[fy][cx]==m:
            truey=fy
            truex=cx
        elif imgarr[cy][fx]==m:
            truey=cy
            truex=fx
        elif imgarr[cy][cx]==m:
            truey=cy
            truex=cx
        centerlist.append((truey,truex))

    return centerlist

def getrowcol(clist):
    clist1=clist.copy()
    clist2=clist.copy()
    listrows = [] #unordered list of rows, rows themselves are lists of centers in the same row
    listcols = [] #unordered list of columns, columns themselves are lists of centers in the same column

    #each pass through the loop finds a row, defined by all beams with y-coord within sep/2 of a random beam's y-coord, we get a list of rows (single row is an array of centers in the same row)
    while clist1:
        randi=random.choice(len(clist1))
        cen=clist1[randi]
        row = []
        for c in clist1:
            if abs(c[0]-cen[0])<sep/2:
                row.append(c)
        listrows.append(row)
        for x in row:
            clist1.remove(x)

    #each pass through the loop finds a column, defined by all beams with x-coord within sep/2 of a random beam's x-coord, we get a list of columns
    while clist2:
        randi=random.choice(len(clist2))
        cen=clist2[randi]
        col = []
        for c in clist2:
            if abs(c[1]-cen[1])<sep/2:
                col.append(c)
        listcols.append(col)
        for x in col:
            clist2.remove(x)

    return listrows,listcols

#lattice positions are (row,col), getlatticepos() works when top row and leftmost column is at least sparsely occupied
def getlatticepos(clist):
    poslist = list()
    #gets position of top left site whether it is occupied or not, by searching for the smallest x value and smallest y value among the list of peaks
    top = min(clist, key=lambda tup: tup[0])
    left = min(clist, key=lambda tup: tup[1])
    ymin = top[0]
    xmin = left[1]
    #gets lattice position based on absolute position relative to top left, and beam separation
    for c in clist:
        poslist.append((round((c[0]-ymin)/sep),round((c[1]-xmin)/sep)))

    return poslist

#returns radii of each beam in form of a tuple list, (radl,radr,radt,radb) 
def getradii(clist, listrows, listcols):
    radlist = list()
    for c in clist:
        #gets left and right radii by comparing beam c to all other beams ctest in its row
        for row in listrows:
            if c in row:
                lmin = xdim
                rmin = xdim
                for ctest in row:
                    if ctest[1]<c[1] and c[1]-ctest[1]<lmin:
                        lmin=c[1]-ctest[1]
                    if ctest[1]>c[1] and ctest[1]-c[1]<rmin:
                        rmin=ctest[1]-c[1]

        #if edge beam, then set the exterior radius to the interior radius
        if lmin==xdim:
            lmin=rmin
        if rmin==xdim:
            rmin=lmin

        radl=lmin//2
        radr=rmin//2
        #gets up and down radii by comparing beam c to all other beams ctest in its column
        for col in listcols:
            if c in col:
                tmin = ydim
                bmin = ydim
                for ctest in col:
                    if ctest[0]<c[0] and c[0]-ctest[0]<tmin:
                        tmin=c[0]-ctest[0]
                    if ctest[0]>c[0] and ctest[0]-c[0]<bmin:
                        bmin=ctest[0]-c[0]

        if tmin==ydim:
            tmin=bmin
        if bmin==ydim:
            bmin=tmin        
        
        radt=tmin//2
        radb=bmin//2

        radlist.append((radl,radr,radt,radb))

    return radlist

#Slices entire image into 2D arrays with one beam per array using radii in each direction, this form works better for the subsequent Gaussian fit, rather than fitting small sections of a larger array
def isolatebeams(clist,rlist):
    beams = list()
    for cen,rad in zip(clist,rlist):
        top = cen[0]-rad[2]
        bottom = cen[0]+rad[3]+1
        left = cen[1]-rad[0]
        right = cen[1]+rad[1]+1
        beams.append(imgarr[top:bottom, left:right])
    return beams

def gauss(x, I, w):
    return I*np.exp(-2.*(x)**2/(w**2))

def getparalist():
    global plist

    clist = getcenters()
    plist = getlatticepos(clist)
    rowlist, collist = getrowcol(clist)
    numrows = len(rowlist)
    rlist = getradii(clist,rowlist,collist)
    beamlist=isolatebeams(clist,rlist)

    #gets list of Gaussian parameters in the form of tuples (index, row, col, xI, xw, yI, yw, col=centerx, row=centery, rsqx, rsqy) for each beam
    paralist = list()
    for beam,rad,cen,pos in zip(beamlist,rlist,clist,plist):
        adjx=0
        #xdist/ydist gives integrated brightness projected along the corresponding axis
        xdist = np.sum(beam, axis=0)
        #sometimes the x value of peak brightness i.e. our "center" is not the same as the x value of peak integrated brightness which is what we want our Gaussian to be centered on, so we make a correction adjx/adjy
        if xdist[rad[0]]<xdist[rad[0]+1]:
            adjx=-1
        if xdist[rad[0]]<xdist[rad[0]-1]:
            adjx=1

        x = np.arange(-1*rad[0]+adjx,rad[1]+1+adjx)
        xpar, xcov = curve_fit(gauss, x, xdist)

        #finds Gaussian parameters for fit in x
        xresiduals = xdist-gauss(x, *xpar)
        xss_res = np.sum(xresiduals**2)
        xss_tot = np.sum((xdist-np.mean(xdist))**2)
        xrsq = 1-(xss_res/xss_tot)
        xpar = absolute(xpar)

        adjy=0
        ydist = np.sum(beam, axis=1)
        if ydist[rad[2]]<ydist[rad[2]+1]:
            adjy=-1
        if ydist[rad[2]]<ydist[rad[2]-1]:
            adjy=1

        y = np.arange(-1*rad[2]+adjy,rad[3]+1+adjy)
        ypar, ycov = curve_fit(gauss, y, ydist)

        #finds Gaussian parameters for fit in y
        yresiduals = ydist-gauss(y, *ypar)
        yss_res = np.sum(yresiduals**2)
        yss_tot = np.sum((ydist-np.mean(ydist))**2)
        yrsq = 1-(yss_res/yss_tot)
        ypar = absolute(ypar)

        center = [cen[1],cen[0]]
        rsq = [xrsq,yrsq]
        #adds index so that the parameter list can be sorted in order of beams from left to right, top to bottom, paralist.sort() sorts by first element in tuple which is index
        index = numrows*pos[0]+pos[1]
        par = (index,pos[0],pos[1],xpar[0],xpar[1],ypar[0],ypar[1],center[0],center[1],rsq[0],rsq[1])
        paralist.append(par)

        #visualizing Gaussian fit for each beam
        """ plt.scatter(x,xdist)
        xmesh = np.linspace(-1*rad[0],rad[1], 200)
        plt.plot(xmesh, gauss(xmesh,xpar[0],xpar[1]))
        plt.show()

        plt.scatter(y,ydist)
        ymesh = np.linspace(-1*rad[2],rad[3], 200)
        plt.plot(ymesh, gauss(ymesh,ypar[0],ypar[1]))
        plt.show() """
    
    paralist.sort()
    return paralist

#eventually just return list of single intensities, how do we get one intensity from the x and y fits?
def getIntenList():
    setup()
    parlist = getparalist()
    #gets dimensions of the beam lattice
    y = max(plist, key=lambda tup: tup[0])
    x = max(plist, key=lambda tup: tup[1])
    rows = y[0]+1
    cols = x[1]+1
    print([(i[3],i[5]) for i in parlist])
    intenlist = [(i[3]+i[5])/2 for i in parlist]
    return np.reshape(intenlist, (rows,cols))
    

#######CODE HERE#######

getIntenList()

#can get lattice dimensions by taking max poslist[0] and max poslist[1] and use those in np.reshape(), since DMDcalib takes in 2D np array as camera input

#######CODE HERE#######
