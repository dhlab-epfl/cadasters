# Cadasters segmentation
Demo for PDM work on venitian cadaster segmentation

This code is a demo for the segmentation of the venitian cadaster. It is a work in progress and is not optimized. Comments and suggestions are welcome.

### Organization
The repo is organized as follows :
* __data/image_cadasters__ contains samples of images to segment. It also contains a full cadaster sheet. It is recommended to select a region and crop it to process. Processing of the full cadaster sheet will be way too long.
* __output_examples__ shows two examples of results
* __cadaster_script.py__ can be launched with command line
* __demo_cadaster_segmentation.ipynb__ allows to play with the parameters

### Usage
* Python Notebook : Run with the given parameters/inputs or try your owns.
* Script : run `python cadaster_script.py -im data/image_cadasters/cadaster_sample1.jpg`

### Outputs
__cropped_polygons__ folder : cropped images of extracted polygons

__digits__ folder : cropped images of digits (binary and original) and json file with the predicted number and its confidence

__Box__ : shows detected labels

__log__ : summary of the processing (parameters used, time for computing, ...)

__geojson__ : coordinates (in the image referential) of the extracted polygons for further integration in web service with geographic coordinates

__polygons__ : shows extracted regions

__prediction class__ : shows how the segments have been classified (3 classes)

__ridge image__ : shows the result of the processing of ridge detection image, this is the input of the flooding algorithm

__merged superpixels__ : shows the result of the merging process (better over-segmented than under-segmented, some corrections are applied later)

### Requirement
* Python 3.5

### Dependencies
* networkx 
* geojson
* json
* cv2
* skimage
* sklearn
* colormath
* uuid
* tensorflow

