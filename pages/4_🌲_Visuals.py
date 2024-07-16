import os
import streamlit as st
import boto3
import ee
import geemap.foliumap as geemap
import leafmap.foliumap as leafmap
import geopandas as gpd
import matplotlib.pyplot as plt
from utils import *

st.set_page_config(layout="wide")

# Load environment variables from .env file
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]
S3_REGION = st.secrets["S3_REGION"]

def download_file_from_s3(file_name, local_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION
    )
    try:
        s3.download_file(BUCKET_NAME, file_name, local_path)
        st.success(f"Successfully downloaded {file_name} from S3")
    except Exception as e:
        st.error(f"Error downloading {file_name}: {str(e)}")

def upload_file_to_s3(local_path, file_name):
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION
    )
    try:
        s3.upload_file(local_path, BUCKET_NAME, file_name)
        st.success(f"Successfully uploaded {file_name} to S3")
    except Exception as e:
        st.error(f"Error uploading {file_name}: {str(e)}")

# Ensure the path to your service account JSON key file is correct
key_file_name = 'jnr-works/jnr-master.json'
local_key_path = 'jnr-master.json'

# Download the service account JSON key file from S3
download_file_from_s3(key_file_name, local_key_path)

# Initialize Earth Engine
service_account = 'jnr-670@jnr-master.iam.gserviceaccount.com'
try:
    credentials = ee.ServiceAccountCredentials(service_account, local_key_path)
    print("initializing")
    ee.Initialize(credentials)
    print("initialized")
except ee.EEException as e:
    st.error(f"Earth Engine initialization failed: {str(e)}")

st.title("Compare rasters")

col1, col2 = st.columns([2, 2])

with col1:
    url1 = st.text_input("Left raster (Recommended)", 'jnr-works/data/adjusted_density.tif')
    local_path1 = url1.split("/")[-1]
    download_file_from_s3(url1, local_path1)
    data1 = st.file_uploader("Select left raster (Very slow)")
    file_path1, layer_name1 = get_filename(data1, local_path1)

with col2:
    url2 = st.text_input("Right raster (Recommended)", 'jnr-works/data/app_adjusted_density.tif')
    local_path2 = url2.split("/")[-1]
    download_file_from_s3(url2, local_path2)
    data2 = st.file_uploader("Select right raster (Very slow)")
    file_path2, layer_name2 = get_filename(data2, local_path2)

m = geemap.Map()
m.split_map(left_layer=file_path1, right_layer=file_path2)
m.to_streamlit()
