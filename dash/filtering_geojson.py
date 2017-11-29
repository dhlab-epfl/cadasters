***REMOVED***
***REMOVED***

import json
***REMOVED***
***REMOVED***
***REMOVED***

files_to_process = glob('/scratch/sofia/updated_cadasters_outputs/*.geojson')

for file in tqdm(files_to_process):
    full_dir, basename = os.path.split(file)
    output_filtered_filename = os.path.abspath(os.path.join(full_dir,
                                                            ''.join(basename.split('_')[:-1] + ['_filt.geojson'])))

    with open(file, 'r') as f:
        data = json.load(f)

    # Remove polygons with less than 3 points
    new_feats = [feat for feat in data['features'] if len(feat['geometry']['coordinates'][0]) > 3]
    data['features'] = new_feats

    # Convert to strings to be able to display it
    for i, p in enumerate(data['features']):
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