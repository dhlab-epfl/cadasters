***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***

# Proj4 format (use for example this site http://epsg.io/3004)
SRC_PROJ = '+proj=tmerc +lat_0=0 +lon_0=15 +k=0.9996 +x_0=2520000 +y_0=0 +ellps=intl +towgs84=-104.1,-49.1,-9.9,0.971,-2.917,0.714,-11.68 +units=m +no_defs'
TRG_PROJ = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

files_to_convert = glob('/scratch/sofia/updated_cadasters_outputs/raw/*.geojson')
for file in tqdm(files_to_convert):
    full_dir, basename = os.path.split(file)
    output_converted_filename = os.path.join(full_dir, '../', '***REMOVED******REMOVED***_wgs84.geojson'.format(basename.split('.')[0]))
    os.system("ogr2ogr -f GeoJSON -s_srs ***REMOVED******REMOVED*** -t_srs ***REMOVED******REMOVED*** ***REMOVED******REMOVED*** ***REMOVED******REMOVED***"
              .format(SRC_PROJ, TRG_PROJ, output_converted_filename, file))