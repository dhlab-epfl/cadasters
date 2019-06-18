***REMOVED***
***REMOVED***

import json
***REMOVED***
***REMOVED***
***REMOVED***
import geojson
from osgeo import ogr
import click

# files_to_process = glob('/scratch/sofia/updated_cadasters_outputs/*.geojson')


@click.command()
@click.argument('files_to_process', nargs=-1)
def filter_geojson(files_to_process):
    for file in tqdm(files_to_process):
        full_dir, basename = os.path.split(file)
        output_filtered_filename = os.path.abspath(os.path.join(full_dir,
                                                                ''.join(basename.split('_')[:-1] + ['_filt.geojson'])))

        with open(file, 'r') as f:
            data = json.load(f)

        # Remove polygons with no points
        new_feats = [feat for feat in data['features'] if len(feat['geometry']['coordinates']) > 0]
        # Remove polygons with less than 3 points
        new_feats = [feat for feat in new_feats if len(feat['geometry']['coordinates'][0]) > 3]
        data['features'] = new_feats

        # Convert to strings to be able to display it
        for i, p in enumerate(data['features']):
            try:
                data['features'][i]['properties']['best_transcription'] = str(p['properties']['best_transcription'])
            except KeyError:
                pass
            data['features'][i]['properties']['transcription'] = str(p['properties']['transcription'])
            data['features'][i]['properties']['score'] = str(p['properties']['score'])

        # Save filterd geojson
        with open(output_filtered_filename, 'w') as f:
            json.dump(data, f)

        # full_dir, basename = os.path.split(output_filtered_filename)
        # output_converted_filename = os.path.join(full_dir, '***REMOVED******REMOVED***_wgs84.geojson'.format(basename.split('.')[0]))
        # # Projection system conversion
        # os.system("ogr2ogr -f GeoJSON -s_srs ***REMOVED******REMOVED*** -t_srs ***REMOVED******REMOVED*** ***REMOVED******REMOVED*** ***REMOVED******REMOVED***"
        #       .format(source_projection, target_projection, output_converted_filename, output_filtered_filename))


***REMOVED***
    filter_geojson()


def convert_multipolygon_to_polygon(files_to_process):
    for filename in files_to_process:

        with open(filename, 'r') as f:
            json_cnt = geojson.load(f)

        export_json = json_cnt.copy()
        export_json['features'] = list()

        for feat in json_cnt['features']:
            new_feat = feat.copy()
            new_feat['geometry'] = ***REMOVED***"coordinates": [], "type": "Polygon"***REMOVED***
            multiploygon = ogr.CreateGeometryFromJson(json.dumps(feat['geometry']))
            for polygon in multiploygon:
                new_feat['geometry'] = json.loads(polygon.ExportToJson())
            export_json['features'].append(new_feat)

        dirname, basename = os.path.split(filename)
        export_filename = os.path.join(dirname, '***REMOVED******REMOVED***_filt.***REMOVED******REMOVED***'.format(*basename.split('.')))
        with open(export_filename, 'w') as f:
            geojson.dump(export_json, f)