#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import click
from tqdm import tqdm
from typing import List
from geojson_processing.geo_info import offset_geojson_file

@click.command
@click.option('--geotif_directory', help='Path to the directory containing the geotif images')
@click.option('--output_dir', help='Output directory for converted geojson files')
@click.argument('filenames', nargs=-1)
def convert_coords_from_geo_to_img(filenames: List[str], geotif_directory: str, output_dir: str):
    for filename in tqdm(filenames):
        offset_geojson_file(filename, geotif_directory, output_dir)


if __name__ == '__main__':
    convert_coords_from_geo_to_img()
