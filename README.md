# HW1: Projective Geometry and Homography

## Instructions
* Late Submission Policy: Five late days total (to be spent wisely)/ 20% off from each extra day.
* Submitting your work: Your code and README (including the proper credits if you use any) should be uploaded to your course website. 
 

## Q1: Affine Rectification (25 points)
Your goal is to generate affinely correct warps for images that are captured through perspective cameras (assume pinhole camera model), with annotations of at least 2 pairs of parallel lines.

**Dataset Preparation**
 1. Find (or capture) `2` images. Ideally capture images that have sufficient perspective. Check the following example:

    | Good Input  | Bad Input |
    | ----------- | ----------- |
    |  <img src="figures/good.jpg" height="200">  | <img src="figures/bad.jpg" height="200"> |

  * Run your code on at least `3` images provided in the `data/q1` folder and *on the images you captured* (`5` images in total). We have provided parallel line annotation for each of them in `annotation/q1_annotation.npy` with an array of `16 x 2` for each image. Each consecutive point represents a line, and each consecutive pair of lines is parallel. We provide a total of 8 lines, out of which the last 4 are held-out lines for Q1.3. 
  Example code to read the annotation: 

  ```python
    with open('q1_annotation.npy','rb') as f:
        q1_annotation = np.load(f, allow_pickle=True)
    points = q1_annotation.item().get('book1')
  ```

**Submission**
1. Input Images
2. Output Images: Rectified images and intermediate images with the color-coded annotation that you used. See the following figures:

    | Input Image | Annotated parallel lines on input image | Affine-Rectified Image |
    | ----------- | ----------- | ----------- |
    |  <img src="figures/tt2.jpg" width="200">  | <img src="figures/tiles5_parallel.png" width="200"> |  <img src="figures/tiles5_affine.png" width="200">  |


3. Evaluate Angles (more precisely, the cosines) of at least another 2 pairs of parallel lines before and after rectification. This will show how far from 1 the cosines are in the original image and how close to correct the final affine rectification is.

    | Before      | After |
    | ----------- | ----------- |
    | 0.7842    | -0.9999    |
    | -0.9999     | 0.9999    |


    | Test lines on Input Image     | Test lines on Affine-Rectified Image |
    | ----------- | ----------- |
    |  <img src="figures/facade_parallel.png" width="200"> | <img src="figures/facade_affine_parallel.png" width="200"> |

4. Briefly describe your implementation (i.e., the algorithm followed with relevant equations and what annotations were used).

## Q2: Metric Rectification (30 points)
Your goal is to generate metrically correct warps for images that are captured through perspective cameras (assume pinhole camera model), with annotations of at least 2 pairs of perpendicular lines, based on your result from Q1.

**Dataset Preparation**
 1. Find (or capture) `2` images. Ideally capture images that have sufficient perspective.
 2. Run your code on at least `3` images provided in the `data/q1` folder and *on the images you captured* (`5` images in total). We have provided perpendicular line annotation for each of them in `annotation/q2_annotation.npy` with an array of `16 x 2` for each image. Each consecutive point represents a line, and each consecutive pair of lines is perpendicular. We provide a total of 8 lines, out of which the last 4 are held-out lines for Q2.3.  The format is the same as `annotation/q1_annotation.npy`. 

**Submission**
1. Input Images
2. Output Images: Rectified images and intermediate images with the color-coded annotation that you used. See the following figures:

    | Input Image | Annotated perpendicular lines on input image | Annotated perpendicular lines on Affine-Rectified Image | Rectified Image |
    | ----------- | ----------- | ----------- | --------- |
    |  <img src="figures/tt2.jpg" width="200">   |  <img src="figures/tiles5_perp.png" width="200">  | <img src="figures/tiles5_affine_perp.png" width="200"> | <img src="figures/tiles5_metric.png" width="200"> |

3. Evaluate Angles (more precisely, the cosines) of at least another 2 pairs of perpendicular lines before and after rectification. This will show how far from 0 the cosines are in the original image and how close to correct the final metric rectification is. Check the following Table and Figure as an example:

    | Before      | After |
    | ----------- | ----------- |
    | -0.66858    | -0.04479    |
    | 0.02118     | -0.00964    |


    | Test lines on Input Image     | Test lines on Metric-Rectified Image |
    | ----------- | ----------- |
    |  <img src="figures/chess1_perp.png" height="200"> | <img src="figures/chess1_metric_perp.png" height="200"> |
4. Briefly describe your implementation (i.e., the algorithm followed with relevant equations and what annotations were used).

## Q3: Planar Homography from Point Correspondences (25 points)
Your goal is to estimate homographies between two images using point correspondences.

**Dataset Preparation**
 1. Find (or capture) `1` pair of images. The image pair should contain a "normal" image with no perspective effects and an image with perspective effects. The goal is to apply a homography to the normal image to place it on top of the other image. Check the following example:

    | Normal Image | Perspective Image |
    | ----------- | ----------- |
    |  <img src="figures/desk-normal.png" height="150">  | <img src="figures/desk-perspective.png" height="150">  |
 2. Run your code on the pair of images provided in the `data/q3` folder as well as `1` additional pair of images that you captured. We have provided annotation of the four corner points in `desk-perspective.png` (in anti-clockwise order) in `annotation/q3_annotation.npy`. The format is the same as `annotation/q1_annotation.npy`. 

**Submission**
1. Input Images
2. Output Images: Warped images and intermediate images with annotation that you used (i.e., corners/point correspondences). See the following figures:

    | Normal Image | Perspective Image | Annotated corners in Perspective Image | Warped and Overlaid Image |
    | ----------- | ----------- | ----------- | ----------- |
    |  <img src="figures/desk-normal.png" height="150">  | <img src="figures/desk-perspective.png" height="150">  |  <img src="figures/desk-corners.png" height="150">  | <img src="figures/desk-homography.png" height="150"> |
3. Biref description of your implementation (i.e., the algorithm followed with relevant equations and what annotations were used).

## Q4: Metric Rectification from Perpendicular Lines (10 points)
In Q2, we generate metrically correct warps of images with annotations of 2 pairs of perpendicular lines based on the result from Q1. In this question, the goal is to metric-rectify the image directly from at least 5 pairs of perpendicular lines. Note you can use more than 5 pairs to get more accurate results.

**Dataset Preparation**
 1. Find (or capture) at least `1` image. Ideally capture an image that has sufficient perspective.
 2. Run your code on at least the `1` image provided in the `data/q1` folder and *on the images you captured*.

**Submission**
1. Input Images
2. Output Images: Rectified images and intermediate images with the color-coded annotation that you used. Annotate the lines that were used. See the following figure:
    | Input Image | Annotated perpendicular lines | Rectified Image |
    | ----------- | ----------- | ----------- |
    |  <img src="figures/tt2.jpg" width="200">   |  <img src="figures/tiles5_anno.png" width="200">  | <img src="figures/tiles5_out.png" width="200">  |
    
3. Angles (more precisely, the cosines) of at least 3 pairs of perpendicular lines before and after rectification.
4. Biref description of your implementation.

## Q5: More Planar Homography from Point Correspondences (10 points)
Try to be creative and do something fun!

For example, in Q3, we attempt to overlay a normal image on top of an image with a perspective effect. You can try to overlay multiple ($\ge 3$) normal images on top of the image with perspective effect:

 | Example output | 
 | ----------- | 
 |  <img src="figures/TimesSquare.png" height="300"> | 

<!-- **Dataset Preparation**
 1. Find (or capture) a `1` image with perspective effect and at least `3` "normal" images. Run your code on this set of images.  -->
 
**Submission**
 1. Input Images
 2. Output Images.
 3. Brief description of your implementation.

## What you can *not* do
* Download any code.
* Use any measurement in the scene provided apriori. (The only information available to the algorithm is the input image and the annotations)
* Use any predefined routines except linear algebra functions, image interpolation, and image warping.
  
## Tips
* You can use the `annotate` function provided in `utils.py` to annotate your own image. It will save the click coordinates and return it as an array.  
* It is a good idea to `assert` with sanity checks regularly during debugging.
* Normalize point and line coordinates.
* Pick the parallel and perpendicular lines carefully.
* Process the transformation matrix properly before the final warping. (e.g., scale, translation, etc.)
* Remember that transformations are estimated up to scale and that you are dealing with Projective Geometry.
* You *may not* use a predefined routine to directly compute homography (e.g., `cv2.findHomography`). However, you *may* use predefined linear algebra/image interpolation libraries (e.g., `np.linalg`, `cv2.warpPerspective`). If you are unsure about what may or may not be used, don't hesitate to ask your TA.
* We provide some utility functions in `utils.py` for reference. Feel free to use it as you like.
* **Start Early and Have Fun!**
