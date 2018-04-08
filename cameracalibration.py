import numpy as np
import os
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import imagetransform

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
        self.M_warp, self.M_unwarp = self._get_warp_matrix()

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

    def undistort(self, img):
        """

        :param image_file: image to undistort
        :return: undistorted image if calibrated
        """
        if (self.calibrated):
            dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            return dst
        else:
            logging.error("Camera is not calibrated, calibrate the camera before using undistort")
            return None

    def _get_warp_matrix(self):
        w, h = 1280, 720
        x, y = 0.5*w, 0.8*h

        self.src = np.float32([
                [200./1280*w, 720./720*h],
                [453./1280*w,547./720*h],
                [835./1280*w,547./720*h],
                [1100./1280*w,720./720*h]
            ])

        self.dst = np.float32([
            [(w-x)/2.,h],
            [(w-x)/2.,0.82*h],
            [(w+x)/2.0,0.82*h],
            [(w+x)/2.,h]
        ])
        M_warp = cv2.getPerspectiveTransform(self.src, self.dst)
        M_unwarp = cv2.getPerspectiveTransform(self.dst, self.src)
        return M_warp, M_unwarp

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M_warp, img_size, flags=cv2.INTER_LINEAR)
        return warped

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        unwarped = cv2.warpPerspective(img, self.M_unwarp, img_size, flags=cv2.INTER_LINEAR)
        return unwarped

    def verify_perspective_transform(self, img):
        polylined = img.copy()
        warped = self.warp(img)
        polylined = cv2.polylines(polylined, np.int32([self.src]), 1, (255,0,0))
        warped = cv2.polylines(warped, np.int32([self.dst]), 1, (255,0,0))
        imagetransform.show_image_pair(polylined, warped, "Original", "Warped")


