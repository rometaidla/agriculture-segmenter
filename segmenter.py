import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import reshape_as_raster, reshape_as_image

import numpy as np

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

import imutils

import cv2

def segment(image_path):
    image = read_image(image_path)
    thresholded_image, gray = otsu_thresholding(image)
    labels = watershed_segment(thresholded_image)
    draw_labels(image, gray, labels)
    return image, thresholded_image

def read_image(image_path):
    raster = rasterio.open(image_path).read()
    image = reshape_as_image(raster)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def otsu_thresholding(image):
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    tresholded_image = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return (tresholded_image, gray)

def watershed_segment(thresholded_image):
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresholded_image)
    localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresholded_image)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresholded_image)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    return labels

# TODO: passing gray should not be needed
def draw_labels(image, gray, labels):
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        for (i, c) in enumerate(cnts):
            # draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(c)
            # cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (255, 0, 0), 2)
            cv2.fillPoly(image, pts=[c], color=(0, 255, 0))