***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

import click
import pandas as pd
import shapely
import re
import numpy as np


def _float2int(unit):
    if isinstance(unit, float) and not pd.isnull(unit):
        return int(unit)
    else:
        return unit

@click.command
@click.option("--csv_filename", help="The file (CSV) containing all the manually annotated geometries.")
@click.option("--export_filename", help="The path to the file (CSV) to export the filtered / cleaned geometries")
def clean_manually_annotated_parcels(csv_filename: str, export_filename: str):
    labels_to_keep = ['WKT', 'best_trans', 'uuid', 'ID']

    manual_parcels = pd.read_csv(csv_filename)
    manual_parcels = manual_parcels[labels_to_keep]

    # Create a geometry from the WKT
    manual_parcels['geometry'] = manual_parcels.WKT.apply(lambda s: shapely.wkt.loads(s)[0])
    manual_parcels = manual_parcels.drop(['WKT'], axis=1)

    # Change float into int
    manual_parcels = manual_parcels.ID.apply(lambda t: _float2int(t))

    # Remove non digits chars and convert transcription to float type
    manual_parcels.ID = manual_parcels.ID.apply(lambda t: re.sub(r"\D", "", str(t)))
    manual_parcels.ID.replace('', np.nan, inplace=True)
    # manual_parcels.ID = manual_parcels.ID.apply(lambda t: float(t) if t != "" else np.nan)

    # Find invalid shapes in groundtruth and remove them also
    invalid_polygons = manual_parcels[~manual_parcels.geometry.apply(lambda s: s.is_valid)]
    print("Removed ***REMOVED******REMOVED*** invalid polygons.".format(len(invalid_polygons)))

    man_corrected = manual_parcels.drop(index=invalid_polygons.index, axis=0)

    man_corrected.to_csv(export_filename)
