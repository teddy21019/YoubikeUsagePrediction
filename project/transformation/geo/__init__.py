import pyproj

LAT_LON_CODE = 4326
TW_COOR_CODE = 3826

tw_transformer = pyproj.Transformer.from_crs(LAT_LON_CODE, TW_COOR_CODE)
lat_lon_to_xy = lambda x,y: tw_transformer.transform(x, y)