import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import reshape_as_image
import numpy as np
import cv2 as cv
import segmenter
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    model_file = 'ndvi_rf.joblib'
    dump(clf, model_file)
    print(f"Model saved to {model_file}.")

def create_classifier():
    clf = load('ndvi_rf.joblib')
    return clf

def load_data():
    image1 = segmenter.read_image('data/06-11-2020/T35VME_20200531_B8.tif')
    image2 = segmenter.read_image('data/06-11-2020/T35VME_20200628_B8.tif')
    image3 = segmenter.read_image('data/06-11-2020/T35VME_20200718_B8.tif')
    images = [image1, image2, image3]

    X = calculate_ndvi_df(images)
    y = create_ground_truth().reshape(-1)
    y[y > 0] = 1

    return X, y

def calculate_ndvi_df(images):
    ndvi = calculate_ndvi(images)
    ndvi = np.array([n.reshape(-1) for n in ndvi])

    ndvi_df = pd.DataFrame(data=ndvi.T, columns=["pic1", "pic2", "pic3"])
    ndvi_df['max'] = ndvi_df.max(axis=1)
    ndvi_df['min'] = ndvi_df.min(axis=1)
    ndvi_df['std'] = ndvi_df.std(axis=1)
    ndvi_df['mean'] = ndvi_df.mean(axis=1)

    return ndvi_df


def create_ground_truth():
    input_dataset = rasterio.open('data/p6llu_piirid.tif')
    raster = input_dataset.read()
    im_in = reshape_as_image(raster)
    th, im_th = cv.threshold(im_in, 220, 255, cv.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    return im_floodfill_inv[25:537, 22:534]  # remove borders


def calculate_ndvi(images):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')

    ndvi_list = []
    # Calculate NDVI
    for image in images:
        band_red = image[:, :, 2].astype(float)
        band_nir = image[:, :, 3].astype(float)
        ndvi = (band_nir - band_red) / (band_nir + band_red)
        ndvi_list.append(ndvi)

    return ndvi_list


