import geojson
from geojson import FeatureCollection
import cv2
import numpy as np

def savePolygons(listPolygon, namesavefile):
     # Object to save
    collectionPolygons = FeatureCollection([poly for poly in listPolygon])

    # Save file
    with open(namesavefile, 'w') as outfile:
        geojson.dump(collectionPolygons, outfile, sort_keys=True)

    return
# ---------------------------------------------------------------------------


def readPolygonfromgeoJSON(filename):
    with open(filename) as f:
        data = geojson.load(f)

    list_polygons = list()
    for feature in data['features']:
        list_polygons.append(feature['geometry']['coordinates'][0])

    return list_polygons
# ---------------------------------------------------------------------------


def crop_polygon(image, polygon):
    """
    Crops the image to get the desired polygon

    :param image: original color image
    :param polygon: polygon to crop
    :return: croped image containing desired polygon
    """

    img_mask = np.zeros(image.shape, 'uint8')
    # Create mask polygon
    cv2.fillPoly(img_mask, polygon, (255, 255, 255))
    # Dilate to have nice polygon
    kernel = np.ones((6, 6), np.uint8)
    img_mask = cv2.dilate(img_mask, kernel)

    # Mask * image
    img_parcel = (img_mask > 0)*image

    # Crop parcel
    tmp, cnt, tmp = cv2.findContours(img_mask[:, :, 0].copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    y, x, h, w = cv2.boundingRect(cnt[0])

    return img_parcel[x:x+w, y:y+h, :]
