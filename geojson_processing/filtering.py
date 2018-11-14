***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

from typing import List, Dict
from shapely.geometry import shape
import geojson
import json
***REMOVED***
import click


def get_valid_shapes(features: List[Dict], verbose: bool=False) -> List[Dict]:
***REMOVED***"

    :param features: list of geojson Feature type objects (shapely.geometry.shape)
    :param verbose: if True will print the invalid Features
    :return: the filtered ``features`` vector
***REMOVED***"

    valid_features = list()
    invalid_count = 0
    for feature in features:
        try:
            shapely_object = shape(feature['geometry'])
            if shapely_object.is_valid:
                valid_features.append(feature)
            else:
                invalid_count += 1
                if verbose:
                    print('- Shape is not valid. (counter invalid: ***REMOVED******REMOVED***)'.format(invalid_count))
                    print(feature)
        except ValueError:
            invalid_count += 1
            if verbose:
                print(' - Geojson shape is not readable. (counter invalid: ***REMOVED******REMOVED***)'.format(invalid_count))
                print(feature)

    print('   Found ***REMOVED******REMOVED*** invalid shapes.'.format(invalid_count))

    return valid_features


def clean_and_export(filename_original: str, export_dir: str) -> None:
***REMOVED***"

    :param filename_original: filename of the original geojson file
    :param export_dir: directory to export the cleaned geojson files
    :return:
***REMOVED***"

    # Load the geojson file content
    with open(filename_original, 'r') as f:
        geojson_content = geojson.load(f)

    # Filter out invalid shapes
    geojson_content['features'] = get_valid_shapes(geojson_content['features'])

    # Export cleaned content
    basename = os.path.basename(filename_original)
    clean_geojson_filename = os.path.join(export_dir, basename)

    with open(clean_geojson_filename, 'w') as f:
        json.dump(geojson_content, f)


# @click.command()
# @click.option('-e', '--export-dir', 'export_dir', help='Folder where the cleaned geojson files will be exported')
# @click.argument('list_filenames', nargs=-1)
def batch_clean_and_export(list_filenames: List[str], export_dir: str):
***REMOVED***"

    :param list_filenames: list of filename of the original geojson files
    :param export_dir: directory to export the cleaned geojson files
    :return:
***REMOVED***"

    if os.path.isdir(export_dir):
        print('Export directory already exists')
    else:
        os.makedirs(export_dir)

    for filename in list_filenames:
        print('Processing ***REMOVED******REMOVED***'.format(filename))
        clean_and_export(filename, export_dir)


***REMOVED***
    batch_clean_and_export()
