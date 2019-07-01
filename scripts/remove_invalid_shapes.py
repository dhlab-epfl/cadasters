***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

import click
from typing import Union, List
from geojson_processing.filtering import batch_clean_and_export


@click.command
@click.option('--export_dir', help='Path to the exporting directory')
@click.argument('geojson_files', nargs=-1)
def remove_invalid(geojson_files: Union[List[str], str],
                   export_dir: str):
    batch_clean_and_export(geojson_files, export_dir)
