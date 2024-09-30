import cv2
import numpy as np



# Algorithm has the following steps:
# 1. Quantize the image with kemans for k = 6. 
# 2. Perform color filtration: Create a mask to find reddish areas; 
# use bitwise_and + mask operator to filter out non-reds.
# 3. Use contour detection to find the COMs of reddish areas 
# 4. Filter out nonsense/outliers based on y-coords
# 5. Perform k-means on these points for k = 16 to sharpen the COMs by approximating centroids
# 6. Decompose the list of points into left + right lines, extrapolate the lines 
# by solving for m and b in y = mx + b for both sides. 
# 7. Sueprimpose the lines onto the img, write it to filesys & done!


# this is a method created to quantize the image using kmeans
# https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
def quantize_image(image, k):

    # 2d archive of all pixels.
    pxls = image.reshape((-1, 3))
    pxls = np.float32(pxls)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # k-means clustering
    ret, labels, centers = cv2.kmeans(pxls, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert to uint8
    centers = np.uint8(centers)

    # Remake original img
    quantized_image = centers[labels.flatten()].reshape(image.shape)

    return quantized_image

# open orig img
image = cv2.imread('red.png')

# Quantization step:

# using k = 6 refines the red cones as much as possible.
quantized_image = quantize_image(image, 6)

image_orig = image # save original image to draw lines on it
image = quantized_image # but we'll mostly be working w/ quantized img

# https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=The%20HSV%20values%20for%20true,10%20and%20160%20to%20180.
# Color filtration step:


# Convert the image to HSV because it's a lot easier to work with for color dxn
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# HSV ranges for red : 
lo_red1 = np.array([0, 100, 20])
hi_red1 = np.array([10, 255, 255])

lo_red2 = np.array([160, 100, 20])
hi_red2 = np.array([179, 255, 255])

# Create masks for red color range1 & range2.
mask1 = cv2.inRange(hsv_image, lo_red1, hi_red1)
mask2 = cv2.inRange(hsv_image, lo_red2, hi_red2)

full_red_mask = mask1 + mask2

# apply a bitwise_and w/ mask to get red regions
# This means we set !(red) to 0, this allows us to only see red portions
red_regions = cv2.bitwise_and(image, image, mask=full_red_mask)

# Contour centroid localization step:

# https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
# find the contours of the now filtered image.

# remark : 
# findContours now only returns contours + hierarchy, no more "im2".
contours, hierarchy = cv2.findContours(full_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find COMs of the contours.

contour_coords = []

# find cX, cY
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        # Calculate the center of the contour
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # remove outliers and nonsensical points... 
        # e.g., the red portion on the door to the left, the red clock characters,
        # and their reflections
        if(cY < 600): continue 
        
        contour_coords.append((cX, cY))



from sklearn.cluster import KMeans

# Refinement step:

# REMARK : This refinement is not gonna be perfect, but all we need 
# is enough clearance to draw some lines thru p_min & p_max ...

# the total number of red cones is 14, but 
# the algorithm works best with k = 16.
contour_coords = np.array(contour_coords)
kmeans = KMeans(n_clusters=16, random_state=0)
kmeans.fit(contour_coords)

centers = kmeans.cluster_centers_

# Equation graphing/line drawing step:

# decomposing centroids to left / right lines.
maskLine1 = centers[:, 0] > 1000
maskLine2 = centers[:, 0] < 1000

pointsLine1 = centers[maskLine1]
pointsLine2 = centers[maskLine2]

maxP1 = pointsLine1[np.argmax(pointsLine1[:, 1])]
maxP2 = pointsLine2[np.argmax(pointsLine2[:, 1])]

minP1 = pointsLine1[np.argmin(pointsLine1[:, 1])]
minP2 = pointsLine2[np.argmin(pointsLine2[:, 1])]

print(pointsLine1)
print(pointsLine2)

# calculating slope + y-intercept
# this is to extrapolate the lines created by the centroids to the 
# entire image

# solving for m, b in y = mx + b
derivative = (maxP2[1] - minP2[1])/(maxP2[0] - minP2[0])
intercept = maxP2[1] - derivative * maxP2[0]

width = 1816


minP2Final = (0, int(intercept))
maxP2Final = (width, int(width * derivative + intercept))

# drawing the line..
cv2.line(image_orig, minP2Final, maxP2Final, color=(0, 0, 255), thickness=3)

# same but for p1.
derivative = (maxP1[1] - minP1[1])/(maxP1[0] - minP1[0])
intercept = maxP1[1] - derivative * maxP1[0]

width = 1816


# BGR (0,0,255) = red line
# I think the thickness in the ex was 3
minP1Final = (0, int(intercept))
maxP1Final = (width, int(width * derivative + intercept))
cv2.line(image_orig, minP1Final, maxP1Final, color=(0, 0, 255), thickness=3)


# write the final answer to filesys
cv2.imwrite("answer.png", image_orig)