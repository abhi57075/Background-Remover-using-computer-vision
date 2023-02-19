# # Importing the libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import rembg

# path of the image
path = 'TEST IMAGES-20221228T215541Z-001/TEST IMAGES/2.jpg'

while True :
            # Read the image
            image = cv.imread(path)

            # Display the original image
            img = cv.resize(image, (1500,750))
            cv.imshow('Original Image', img)

            print(f'Image shape is : {img.shape}')

            # Select ROI manually
            r = cv.selectROI("select the ROI manually", img) # x,y,length,breadth
            print(f'Co ordinates of cropped section is {r}')
            
            # Crop image
            cropped_image = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            print(f"Cropped Image shape is : {cropped_image.shape}")

            # Display cropped image
            cv.imshow("Cropped image", cropped_image)

            # Removing the background of the cropped image
            rbgimg = rembg.remove(cropped_image)
            print(f"Rbg image shape is : {rbgimg.shape}")

            # Display the rbgimg
            cv.imshow("Background removed img", rbgimg)

            # Drawing outline over the rbgimg
            imgray = cv.cvtColor(rbgimg, cv.COLOR_BGR2GRAY)
            imgray = cv.bitwise_not(imgray)
            print(f"Image gray shape is {imgray.shape}")

            imgray = cv.GaussianBlur(imgray, (5,5), cv.BORDER_DEFAULT) 

            ret, thresh = cv.threshold(imgray, 252, 255, 0)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            cv.drawContours(rbgimg, contours, -1, (0,255,0), 2)
            cv.imshow("Contours drawn", rbgimg)
            print(f"Rbg image shape is : {rbgimg.shape}")

            rbgimg = rbgimg[:,:,:3]
            processed_img = cv.bitwise_or(cropped_image, rbgimg)
            cv.imshow("Processed Image", processed_img)

            img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = processed_img
            cv.imshow("Final Image", img)

            cv.waitKey(0)

            if cv.waitKey(1) & 0xFF == ord('q'): # when the key 'q' is pressed on the keyboard there will be a delay of 1ms and the video will stop playing
                cv.destroyAllWindows()
                break
            else :
                cv.destroyWindow('Final Image')
                cv.destroyWindow('Processed Image')
                cv.destroyWindow('Contours drawn')
                cv.destroyWindow('Background removed img')
                cv.destroyWindow('Cropped image')


            



