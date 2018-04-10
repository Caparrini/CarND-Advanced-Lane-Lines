from cameracalibration import CalibratedCamera
import logging
import cv2
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)
import glob
import os
import imagetransform
import linesearch
import numpy as np


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
images_undistorted = [cal_cam.undistort(x) for x in images]
plotImages(images, images_undistorted)


#3)Use color transforms, gradients, etc., to create a thresholded binary image.
straight_image = cv2.imread("test_images/straight_lines1.jpg")
plt.imshow(straight_image)
plt.show()
warped = cal_cam.warp(straight_image)
plt.imshow(warped, cmap='gray')
plt.show()

transformed = imagetransform.pipeline(straight_image.copy())
plt.imshow(transformed, cmap='gray')
plt.show()

#4)Apply a perspective transform to rectify binary image ("birds-eye view").
warped = cal_cam.warp(transformed)
plt.imshow(warped, cmap='gray')
plt.show()

# Verify Perspective transform dst and src points
cal_cam.verify_perspective_transform(straight_image)

#5)Detect lane pixels and fit to find the lane boundary.
filtered = linesearch.slide_window_search(warped)

binary_filtered = imagetransform.image2binary(filtered)
plt.imshow(binary_filtered, cmap='gray')
plt.show()

warped_polygon, _ = imagetransform.get_polyfill_image(binary_filtered)
plt.imshow(warped_polygon, cmap='gray')
plt.show()

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
    image = cal_cam.undistort(image)
    transformed = imagetransform.new_pipeline(image)
    warped = cal_cam.warp(transformed)
    filtered = linesearch.slide_window_search(warped)
    binary_filtered = imagetransform.image2binary(filtered)
    warped_polygon, curv = imagetransform.get_polyfill_image(binary_filtered)
    unwarped_polygon = cal_cam.unwarp(warped_polygon)
    augmented_original = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1, unwarped_polygon, 0.5,
                                         0.0)  # overlay the orignal road image with window results

    cv2.putText(augmented_original, 'Radius of Curvature: ' + str(round(curv)) + '(m)', (50, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return augmented_original



current_lane_fitter = linesearch.Lane_fitter()


def reset_lane_fitter():
    global current_lane_fitter
    current_lane_fitter = linesearch.Lane_fitter()

def map_line(img):
    global current_lane_fitter
    global cal_cam

    img = cal_cam.undistort(img)
    transformed = imagetransform.new_pipeline(img)
    warped = cal_cam.warp(transformed)

    window_width = 35
    window_height = 80
    curve_centers = linesearch.Tracker(Mywindow_width=45, Mywindow_height=80, Mymargin=40, Mysmooth_factor=100)
    detected_lanes, leftx, rightx = curve_centers.detect_lines(warped)
    left_lane, inner_lane, right_lane, curve_radians, center_diff, side_pos = current_lane_fitter.fit_lanes(warped,
                                                                                                            window_width,
                                                                                                            window_height,
                                                                                                            leftx,
                                                                                                            rightx)

    result = imagetransform.draw_road_lanes(img, left_lane, inner_lane, right_lane, cal_cam)

    cv2.putText(result, 'Radius of Curvature: {0} meters'.format(str(round(curve_radians, 2))), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Distance from center: {0} meters'.format(str(abs(round(center_diff, 2)))), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return transformed, warped, detected_lanes, result


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
        reset_lane_fitter()
        _,_,_,img = map_line(img)
        plt.imshow(img)
    plt.show()

test_process()


def process_frame(image):
    filtered, warped, detected_lanes, result = map_line(image)
    return result

def process_frame2four(image):
    filtered, warped, detected_lanes, result = map_line(image)
    vis_left = np.concatenate((cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB), detected_lanes), axis=0)
    vis_right = np.concatenate((cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB), result), axis=0)
    vis = np.concatenate((vis_left, vis_right), axis=1)
    return vis

project_video_input = 'project_video.mp4'
project_video_output = os.path.join("output_video","project_video.mp4")

reset_lane_fitter()

clip = VideoFileClip(project_video_input)
project_clip = clip.fl_image(process_frame)
project_clip.write_videofile(project_video_output, audio=False)

reset_lane_fitter()

clip4 =  VideoFileClip(project_video_input)
video_clip4 = clip4.fl_image(process_frame2four)
video_clip4.write_videofile(project_video_output + "_process.mp4", audio = False)


