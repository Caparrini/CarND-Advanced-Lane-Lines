import numpy as np
import os
import cv2
import glob
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


class CalibratedCamera(object):
    """
    Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
    """
    def __init__(self, input_calibration_folder):
        """

        :param input_calibration_folder: Which contains the calibration images with format calibration*.jpg
        """
        self.calibrated = False
        self.input_calibration_folder = input_calibration_folder

    def isCalibrated(self):
        return self.calibrated

    def _plotImages(self, raw_images, corners_images):

        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            if(i%2==0):
                img = corners_images.pop()
            else:
                img = raw_images.pop()
            plt.imshow(img)
        plt.show()

    def calibrate(self):
        logging.info("Calibrating camera")
        # Number of inside corners in axis x
        nx = 9
        # Number of inside corners in axis y
        ny = 6

        nimages_with_corners = 0
        # read images
        images_files = glob.glob(os.path.join(self.input_calibration_folder, "calibration*.jpg"))
        logging.debug("There are {0} images for calibration".format(str(len(images_files))))
        objpoints = []  # 3D points in real world
        imgpoints = []  # 2D points in the image
        raw_images = []
        corners_images = []
        bad_images = []

        for image in images_files:
            # Read and plot image
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            objp = np.zeros((ny * nx, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x,y coordinates

            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                raw_images.append(img.copy())
                nimages_with_corners += 1
                imgpoints.append(corners)
                objpoints.append(objp)
                # Draw and display corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                corners_images.append(img)
            else:
                bad_images.append(img)

        logging.debug("Succesfully corners found in {0} of {1} images".format(str(nimages_with_corners),
                                                                              str(len(images_files))))
        img = cv2.imread(images_files[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                               gray.shape[::-1], None, None)
        logging.debug("Ret value is {0}".format(str(ret)))
        self._plotImages(raw_images, corners_images)
        self.calibrated = True
        logging.info("Calibration done!")

    def undistort(self, image_file):
        """

        :param image_file: image to undistort
        :return: undistorted image if calibrated
        """
        if (self.calibrated):
            img = cv2.imread(image_file)
            dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            return dst
        else:
            logging.error("Camera is not calibrated, calibrate the camera before using undistort")
            return None
