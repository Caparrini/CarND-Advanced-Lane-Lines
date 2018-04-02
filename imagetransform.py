import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import glob
import os

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def plot_comparation(img1,img2,img1_name='img1',img2_name='img2'):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(img1_name, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(img2_name, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def image2binary(img, thr=0):
    binary_image = np.zeros_like(img)
    binary_image[(img > thr)] = 1
    return binary_image

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = new_img[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel>thresh[0]) & (s_channel<=thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def binary_and(binary_img_1, binary_img_2):
    """
    Return intersection of two binary images
    """
    if binary_img_1.shape == binary_img_2.shape:
        output = np.where(np.logical_and(binary_img_1,binary_img_2)==True, 1., 0.)
        return output
    else:
        return None

def binary_or(binary_img_1, binary_img_2):
    """
    Return intersection of two binary images
    """
    if binary_img_1.shape == binary_img_2.shape:
        output = np.where(np.logical_or(binary_img_1,binary_img_2)==True, 1., 0.)
        return output
    else:
        return None

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def split_image(image):
    """
    Split image into two halfs cutting by a vertical line at the center
    """
    imshape = image.shape
    # left mask
    vertices_left = np.array([[(0, imshape[0]), (0, 0),
                               (imshape[1] / 2, 0), (imshape[1] / 2, imshape[0])]], dtype=np.int32)
    vertices_right = np.array([[(imshape[1] / 2, imshape[0]), (imshape[1] / 2, 0),
                                (imshape[1], 0), (imshape[1], imshape[0])]], dtype=np.int32)
    imleft = image2binary(region_of_interest(image, vertices_left))
    imright = image2binary(region_of_interest(image, vertices_right))

    return imleft, imright

def get_polyfitline(binary_image):
    coordinates = np.where(binary_image == 1)
    y_range = coordinates[0]
    x_range = coordinates[1]
    fit_line = np.polyfit(y_range, x_range, 2)
    return fit_line, curvature(x_range, y_range)

def get_polyfitlines(binary_image):
    imleft, imright = split_image(binary_image)
    left_line, left_curvature = get_polyfitline(imleft)
    right_line, right_curvature = get_polyfitline(imright)
    return left_line, right_line, np.mean((left_curvature, right_curvature))

def get_polyfill_image(binary_image):
    logging.debug("Getting polyfit lines for left and right images.")
    left_line, right_line, curv = get_polyfitlines(binary_image)

    binary_l = np.zeros_like(binary_image, dtype=np.uint8)

    ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
    plotx_l = left_line[0] * ploty ** 2 + left_line[1] * ploty + left_line[2]
    plotx_r = right_line[0] * ploty ** 2 + right_line[1] * ploty + right_line[2]

    line_points_l = np.column_stack((plotx_l, ploty))
    line_points_r = np.column_stack((plotx_r, ploty))
    line_points = np.concatenate((line_points_l, line_points_r[::-1], line_points_l[:1]))

    cv2.fillPoly(binary_l, np.int32([line_points]), color=255)

    polygon = np.dstack((np.zeros(binary_image.shape), binary_l, np.zeros(binary_image.shape))).astype('uint8')

    print("Curvature: {0}".format(curv))

    return polygon

def curvature(x, y):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 50 / 720  # meters per pixel in y dimension
    xm_per_pix = 7.4 / 1280  # meters per pixel in x dimension

    y_eval = np.max(y)

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

    # Radio of the curvature
    curve_radio = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curve_radio

def pipeline(img):
    logging.debug("Applying pipeline to get binary image...")
    sobel = abs_sobel_thresh(img,'x',30,60)
    hls = image2binary(hls_select(img, (160, 255)))

    output = binary_or(hls, sobel)
    return image2binary(output)

def test_pipeline(folder="test_images"):
    images_files = glob.glob(os.path.join(folder, "test*.jpg"))[2:]
    images = [cv2.imread(x) for x in images_files]
    #images_undistorted = [cal_cam.undistort(x) for x in images_files]
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 4
    for i in range(1, columns * rows + 1, 2):
        fig.add_subplot(rows, columns, i)
        img = images.pop()
        plt.imshow(img)
        fig.add_subplot(rows, columns, i+1)
        img = pipeline(img)
        plt.imshow(img)
    plt.show()

