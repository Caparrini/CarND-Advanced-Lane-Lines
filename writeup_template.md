## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/CalibrationImages.png "Calibration images"
[image2]: ./output_images/undistort.png "Undistorted images"
[image3]: ./output_images/original.png "Original image"
[image4]: ./output_images/warped.png "Warped image"
[image5]: ./output_images/processed.png "Processed image"
[image6]: ./output_images/processed_warped.png "Processed image"
[image7]: ./output_images/processed_warped_filtered.png "Processed image"
[image8]: ./output_images/processed_warped_polynom.png "Processed image"
[image9]: ./output_images/processed_warped_polynom_unwarped.png "Processed image"
[image10]: ./output_images/warpverification.png "Original - Warp / src - dst"
[image11]: ./output_images/Processed_image.png "Processed image"
[image12]: ./output_images/test_images_processed.png "Processed test images"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for calibration is in the class CalibratedCamera in file `cameracalibration.py`.  

The class has a method `calibrate()` which use the images named calibration*.jpg in the calibration folder given at the creation of
the object. These images are chessboards photos taken by the camera. 

9 corners in axis x and 6 axis in axis y are searched using `cv2.findChessBoardCorners()` method, the useful images are used and the others are discarded.

![alt text][image1]

This object calibrated has the method `undistort()` which receives an image and returns the undistorted imaged if the camera is calibrated.
 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The undistort process is aplied through the calibration camera object mentioned early. In the next image three examples of images are shown: without calibration (left) and calibrated (right).
![alt text][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at method new_pipeline() at `imagetransform.py`).  Here is an example of the pipeline applied to a test image:

![alt text][image5]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the `cameracalibration.py` module. The class CalibratedCamera 
includes a function called `warp()` and `unwarp()`.  The `warp()` 
function takes as inputs an image (`img`) and use the src and dst points defines in the code to make the warp. 
 I chose the hardcode the source and destination points in the following manner:

```python
        w, h = 1280, 720
        x, y = 0.5*w, 0.8*h
        src = np.float32([
                [200./1280*w, 720./720*h],
                [453./1280*w,547./720*h],
                [835./1280*w,547./720*h],
                [1100./1280*w,720./720*h]
            ])
        dst = np.float32([
            [(w-x)/2.,h],
            [(w-x)/2.,0.82*h],
            [(w+x)/2.0,0.82*h],
            [(w+x)/2.,h]
        ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 700      | 320, 720        | 
| 453, 547      | 320, 590      |
| 835, 547     | 960, 590      |
| 1100, 720      | 960, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test 
image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]

![alt text][image3]
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I detected lane lines from the warped image using a sliding-window-based tracking method from Udacity video tutorial. This method 
first consist in summing the vertical pixel values in the image, then making a histogram starting from the base of the image where two peaks are identified which should be 
the position of the lane lines.

![alt text][image7]


Later we do movements of that peaks searching the next pixels that seems to be part of the lane lines. This is tackle using the `Tracker()` 
class at the `linesearch.py` module.

Then These points are fitted into a polyline curve. At one frame it is used the MSE of the fitted polylines using the previous fitted polylines. 
Finally the values are generated for each line at the current frame and are drawn into the original image.

![alt text][image8]

![alt text][image9]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in the `fit_lanes()` method into `linesearch.py` module. Beginning with a second order polynomial fitting to the left lane which is then used to calculate the angle of this lane curvature.

Whe need the middle of the x's values of both curves we calculate the position of the vehicle. Comparing it with the center of the image (center of the camera).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step  `main.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image11]

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/mNhPY0AigPs)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project is complex to tackle due to the pipeline which firstly process each frame with gradients and color filters depends on the 
concrete images that are being used. For example with snow the pipeline will not be very useful and similar things happens with changes in the color of the road. Some roads which are being fixed have 
old lane lines on the ground, this pipeline will fail to detect that.

To make it more robust one approach could be to have methods for different environmental conditions, 
and beginning the pipeline detecting the concrete conditions under the vehicle is.

One problem I found out is at processing the video. There are situation when an 'outlier' frame gives false lane lines but it is tackle taking a history of previous lane lines.

Finally one technique could be used is to force the lane 
lines to have a 'shape' among a set defined among 'well-shaped' lane lines, so when a lane line is going to be detected incorrectly it is force to 
have an acceptable shape due to other lane lines fitted previously in other situations. This would be useful when shadows cover part of the lane lines or when 
a lane lines of dots is being searched. 
