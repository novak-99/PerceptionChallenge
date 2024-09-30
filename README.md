# Perception Challenge 


## Algo Used 

# 1. Quantize the image with kemans w/ k = 6. This is to cluster the "red" of the image.
# 2. Perform color filtration: Create a mask to find reddish areas; use bitwise_and + mask operator to filter out non-reds.
# 3. Use contour detection to find the centers of reddish areas 
# 4. Filter out nonsense/outliers based on y-coords
# 5. Perform k-means on the remaining points w/ k = 16 to sharpen the COMs by approximating centroids
# 6. Decompose the list of points into left + right lines, extrapolate the lines by solving for m and b in y = mx + b for both left and right lines so they extend to the full image.
# 7. Superimpose the lines onto the img, write the result to filesys & done!

## Libraries used 

# 1. OpenCV for color quantization (first k-means), color filtration, contour detection, image i/o
# 2. Numpy for arrays, type conversions
# 3. sklearn for performing the second k-means on the tabular set of points.

## Results

![result](answer.png)

## Stuff that failed 

In addition to the above mentioned algorithm (which worked), I also tried to use OpenCV's template matching algorithm to find the cone centroids, but this didn't really work. The cones in the image were just too different to try to detect them all with a single screenshot of any one of them.