# Cadasters segmentation
Demo for PDM work on venitian cadaster segmentation

This code is a demo for the segmentation of the venitian cadaster. It is a work in progress and is not optimized at all. Comments and suggestions are welcome.

### Organization
The repo is organized as follows :
* __data/image_cadasters__ contains samples of images to segment. It also contains a full cadaster sheet. It is recommended to select a region and crop it to process. Processing of the full cadaster sheet will be way too long.
* __output_examples__ shows two examples of results
* __cadaster_script.py__ can be launched with command line
* __demo_cadaster_segmentation.ipynb__ allows to play with the parameters

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

