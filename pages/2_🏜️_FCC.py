import streamlit as st
st.set_page_config(layout="wide")

import geemap.foliumap as geemap
import leafmap.foliumap as leafmap
import geopandas as gpd
from utils import download_file_from_s3, google_cloud_init
from fcc import *
import os

# This must be the first Streamlit command

# Ensure the path to your service account JSON key file is correct
service_account = 'jnr-670@jnr-master.iam.gserviceaccount.com'
key_file_name = 'jnr-works/jnr-master.json'
local_key_path = 'jnr-master.json'

# Download the service account JSON key file from S3
download_file_from_s3(key_file_name, local_key_path)

# Initialize Earth Engine
google_cloud_init(service_account, local_key_path)

st.title("Forest change (Hansen Dataset)")

col1, col2 = st.columns([2, 2])

with col1:
    shapefile_s3_path = st.text_input(
        "Enter path to the jurisdiction shapefile in S3",
        r"jnr-works/data/vichada_boundary1566.shp",
    )
    shapefile_base = os.path.splitext(shapefile_s3_path)[0]
    shapefile_local_path = shapefile_s3_path.split("/")[-1]
    download_file_from_s3(shapefile_base + ".shp", shapefile_local_path + ".shp")
    download_file_from_s3(shapefile_base + ".shx", shapefile_local_path + ".shx")
    download_file_from_s3(shapefile_base + ".dbf", shapefile_local_path + ".dbf")
    # Optional: Download any other associated files like .prj
    download_file_from_s3(shapefile_base + ".prj", shapefile_local_path + ".prj")
    
with col2:       
    outfile = st.text_input(
        "Enter output path for the FCC raster",
        "fcc.tif",
    )

layer_name = shapefile_local_path.split("/")[-1].split(".")[0]
gdf = gpd.read_file(shapefile_local_path + ".shp")
lon, lat = leafmap.gdf_centroid(gdf)
overlapping_boundaries, shapefile_ee = get_fao_data(shapefile_local_path + ".shp")

if overlapping_boundaries is not None:
    region = overlapping_boundaries.geometry().bounds().getInfo()['coordinates']

    fcc = getdata(overlapping_boundaries)
    geemap.ee_export_image(fcc, filename=outfile, scale=90, region=region)

    m = geemap.Map(zoom=4)
    m.addLayer(shapefile_ee, {}, "Shapefile")
    m.addLayer(fcc, {}, "Hansen")
    m.centerObject(fcc)
    m.to_streamlit()
