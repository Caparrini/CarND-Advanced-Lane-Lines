from cameracalibration import CalibratedCamera
import logging
import cv2
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)
import glob
import os
import imagetransform
import linesearch


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
straight_image = cv2.imread("test_images/straight_lines1.jpg")
transformed = imagetransform.pipeline(straight_image.copy())
plt.imshow(transformed, cmap='gray')
plt.show()

#4)Apply a perspective transform to rectify binary image ("birds-eye view").
warped = cal_cam.warp(transformed)
plt.imshow(warped, cmap='gray')
plt.show()

#5)Detect lane pixels and fit to find the lane boundary.
filtered = linesearch.slide_window_search(warped)

binary_filtered = imagetransform.image2binary(filtered)
plt.imshow(binary_filtered, cmap='gray')
plt.show()

warped_polygon = imagetransform.get_polyfill_image(binary_filtered)
plt.imshow(warped_polygon, cmap='gray')
plt.show()

#6)Determine the curvature of the lane and vehicle position with respect to center.

#7)Warp the detected lane boundaries back onto the original image.
unwarped_polygon = cal_cam.unwarp(warped_polygon)
plt.imshow(unwarped_polygon, cmap='gray')
plt.show()

augmented_original = cv2.addWeighted(straight_image, 1, unwarped_polygon, 0.5, 0.0)  # overlay the orignal road image with window results
plt.imshow(augmented_original, cmap='gray')
plt.show()
#8)Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

from moviepy.editor import VideoFileClip
history = []
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    transformed = imagetransform.pipeline(image)
    warped = cal_cam.warp(transformed)
    filtered = linesearch.slide_window_search(warped)
    binary_filtered = imagetransform.image2binary(filtered)
    warped_polygon = imagetransform.get_polyfill_image(binary_filtered)
    unwarped_polygon = cal_cam.unwarp(warped_polygon)
    augmented_original = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1, unwarped_polygon, 0.5,
                                         0.0)  # overlay the orignal road image with window results

    return augmented_original

def test_process(folder="test_images"):
    images_files = glob.glob(os.path.join(folder, "test*.jpg"))
    images_files.extend(glob.glob(os.path.join(folder,"straight_lines*.jpg")))
    images = [cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB) for x in images_files]
    #images_undistorted = [cal_cam.undistort(x) for x in images_files]
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 4
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        img = images.pop()
        img = process_image(img)
        plt.imshow(img)
    plt.show()

test_process()

project_video_input = 'project_video.mp4'
project_video_output = os.path.join("output_video","project_video.mp4")

clip1 = VideoFileClip(project_video_input)
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_video_output, audio=False)


