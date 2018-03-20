from cameracalibration import CalibratedCamera
import logging
import cv2
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)
import glob
import os


#1)Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
cal_cam = CalibratedCamera("camera_cal")
cal_cam.calibrate()

#2)Apply a distortion correction to raw images.
def plotImages(raw_images, corners_images):
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 3
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if (i % 2 == 0):
            img = corners_images.pop()
        else:
            img = raw_images.pop()
        plt.imshow(img)
    plt.show()

images_files = glob.glob(os.path.join("camera_cal","calibration*.jpg"))[:3]
images = [cv2.imread(x) for x in images_files]
images_undistorted = [cal_cam.undistort(x) for x in images_files]
plotImages(images, images_undistorted)


#3)Use color transforms, gradients, etc., to create a thresholded binary image.

#4)Apply a perspective transform to rectify binary image ("birds-eye view").

#5)Detect lane pixels and fit to find the lane boundary.

#6)Determine the curvature of the lane and vehicle position with respect to center.

#7)Warp the detected lane boundaries back onto the original image.

#8)Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

