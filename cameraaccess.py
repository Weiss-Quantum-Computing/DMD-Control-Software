#more code examples found at https://github.com/basler/pypylon/tree/master/samples
from pypylon import pylon
from pypylon import genicam 
import sys 
import numpy as np

##attempting to first connect to the camera to take a snapshot: 
##selecting a number of images 
number_of_images=1

#exitcode of sample application 
exitcode=0


### BOOT-UP SEQUENCE ####
try: 
    #initializing the camera here 
    camera=pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()


    #modifying specs of camera in this section#
    #let's try and change the mono
    print(camera.PixelFormat.Symbolics)
    camera.PixelFormat.SetValue("Mono10") # every pixel should now be stored as a 10 bit integer value
    print(camera.PixelFormat.GetValue())
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(5000)#microseconds, lowest value is 81 microseconds, previously set to 4250    


## STARTING TO RECORD IMAGES, AS WELL AS PROCESSING ###
    #let's begin by grabbing some images, currently set to number__of_images: 
    camera.StartGrabbingMax(number_of_images)

    #It will stop grabbing automatically once it reaches the maximum capacity 

    while camera.IsGrabbing(): 
        
        #what are we doing in this loop: First, we grab the output of the camera and save it under
        # a varaible named grabResult. If an image is successfully saved, we record some information about
        #the photo (widght, height, etc for testing), as well as store the values in an array, named img.
        #We can change what type of photo the image is saved as (I found that using TIFF or PNG was best), 
        # and give it a unique name that is saved in this applications photo
        #If we want more than one photo, the images are saved in a buffer, with number_of_images being the
        #control for how many images are stored in the buffer.


        #How the buffer works is that it stores the images in a queue; an image's information is sent, the 
        #resepctive tasks are given, then the subsequent image is loaded in. How many images saved is 
        #controlled by number_of images
        for picture in range (number_of_images): 
            grabResult=camera.RetrieveResult(5000,pylon.TimeoutHandling_ThrowException)
            
        #now we're checking if we've successfully saved an image 
            if grabResult.GrabSucceeded():
            ##checking the by verifying image descriptions
                    print("Image Width:", grabResult.Width)
                    print("Image Height",grabResult.Height)
                    img=grabResult.Array
                    print("Highest Grey  Pixel;:",np.amax(img))
                    

        ## now onto saving the image 
                    image=pylon.PylonImage()
                    image.AttachGrabResultBuffer(grabResult)
                    filename="trial2.png"
                    image.Save(pylon.ImageFileFormat_Png,filename)

            else: 
                    print("Oh no an error occured:",grabResult.ErrorCode)
        grabResult.Release()
    camera.Close()
 
##I think this should be handling an error, as well as closing out the application##

except genicam.GenericException as e: 
    print("An error has occured")
    print(e.GetDescription())
    exitcde=1

sys.exit(exitcode)
