#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import click
import pandas as pd
from geojson_processing.filtering import clean_manually_annotated_parcels


def _float2int(unit):
    if isinstance(unit, float) and not pd.isnull(unit):
        return int(unit)
    else:
        return unit

@click.command
@click.option("--csv_filename", help="The file (CSV) containing all the manually annotated geometries.")
@click.option("--export_filename", help="The path to the file (CSV) to export the filtered / cleaned geometries")
def clean_annotated_parcels(csv_filename: str, export_filename: str):

    manual_parcels = pd.read_csv(csv_filename)
    man_corrected = clean_manually_annotated_parcels(manual_parcels)
    man_corrected.to_csv(export_filename)
