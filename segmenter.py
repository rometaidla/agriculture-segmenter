import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import reshape_as_image
import numpy as np
from skimage import feature
from matplotlib import pyplot as plt
import random
import cv2 as cv


def read_write_test(input_path, output_path):
    with rasterio.open(input_path) as input_dataset:
        raster = input_dataset.read()
        image = reshape_as_image(raster)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segment_mask, edges = segment(image)
        result = np.append(raster, [segment_mask], axis=0)

        profile = input_dataset.profile
        profile.update(count=4)

        with rasterio.open(output_path, 'w', **profile) as output_dataset:
            output_dataset.write(result)


def segment(image):
    # to black and white
    bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # detect edges
    edges = feature.canny(bw, sigma=0.2)
    edges = edges.astype('uint8')

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv.threshold(edges, 0.4, 1.0, cv.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    dist = cv.bitwise_not(dist) - 254

    dist = cv.distanceTransform(dist, cv.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    # cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)

    # watershed
    dist_8u = dist.astype('uint8')
    # Find total markers
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    # cv.imshow('Markers', markers*10000)
    labels = cv.watershed(image, markers)
    # mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    # plt.imshow(mark)
    # Generate random colors
    colors = []
    cmap = plt.cm.get_cmap('gray', len(contours))
    for i_label, contour in enumerate(contours):
        # colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
        segment_color = cmap(i_label, alpha=0, bytes=True)
        colors.append((segment_color[0].item(), segment_color[1].item(), segment_color[2].item()))

    # print(colors)
    random.shuffle(colors)
    # print(colors)
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]

    segments = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    return segments, edges


def read_image(image_path):
    raster = rasterio.open(image_path).read()
    image = reshape_as_image(raster)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image
