import streamlit as st
import geemap.foliumap as geemap
import geemap.colormaps as cm
import ee
from utils import download_file_from_s3, google_cloud_init

st.set_page_config(layout="wide")

# Define necessary functions
def categorize_changes(image1, image2, image3):
    condition1 = image1.eq(1).And(image2.eq(0)).And(image3.eq(0))
    condition2 = image1.eq(1).And(image2.eq(1)).And(image3.eq(0))
    condition3 = image1.eq(1).And(image2.eq(1)).And(image3.eq(1))
    new_image = ee.Image(0).where(condition1, 1).where(condition2, 2).where(condition3, 3)
    new_image = new_image.where(new_image.eq(0), 0)
    return new_image

def get_hansen_data(year):
    if year == "2014":
        nyear = 13
    elif year == "2019":
        nyear = 18
    else:
        nyear = 22

    gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    treecover = gfc.select(["treecover2000"])
    lossyear = gfc.select(["lossyear"])
    forest2000 = treecover.gte(10)
    loss = lossyear.gte(1).And(lossyear.lte(nyear))
    forest = forest2000.where(loss.eq(1), 0)
    return forest

def getdata(boundry):
    gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").clip(boundry)
    treecover = gfc.select(["treecover2000"])
    lossyear = gfc.select(["lossyear"])
    forest2000 = treecover.gte(10)
    loss1 = lossyear.gte(1).And(lossyear.lte(13))
    loss2 = lossyear.gte(1).And(lossyear.lte(18))
    loss3 = lossyear.gte(1).And(lossyear.lte(23))
    forest2014 = forest2000.where(loss1.eq(1), 0)
    forest2019 = forest2000.where(loss2.eq(1), 0)
    forest2024 = forest2000.where(loss3.eq(1), 0)
    change_raster_hansen = categorize_changes(forest2014, forest2019, forest2024)
    return change_raster_hansen

def get_fao_data(url):
    shapefile_ee = geemap.shp_to_ee(url)
    user_shapefile = ee.FeatureCollection(shapefile_ee)
    buffered_shapefile = user_shapefile.map(lambda feature: feature.buffer(-1000))
    fao_country_boundary = ee.FeatureCollection('FAO/GAUL/2015/level0')
    fao_boundaries = ee.FeatureCollection('FAO/GAUL/2015/level1')
    fao_l2_boundaries = ee.FeatureCollection('FAO/GAUL/2015/level2')
    overlapping_country_boundary = fao_country_boundary.filterBounds(buffered_shapefile.geometry())
    overlapping_boundaries = fao_boundaries.filterBounds(buffered_shapefile.geometry())
    buffered_l2_shapefile = overlapping_boundaries.map(lambda feature: feature.buffer(-1000))
    overlapping_l2_boundaries = fao_l2_boundaries.filterBounds(buffered_l2_shapefile.geometry())
    merged_boundary = ee.FeatureCollection(overlapping_boundaries.geometry().dissolve())
    merged_area = merged_boundary.geometry().area().multiply(100).divide(1000000)
    country_area = overlapping_country_boundary.geometry().area().multiply(100).divide(1000000)
    if merged_area.getInfo() < 2500000:
        buffered_shapefile = overlapping_boundaries.map(lambda feature: feature.buffer(1000))
        buffered_country_boundary = overlapping_country_boundary.map(lambda feature: feature.buffer(-1000))
        overlapping_boundaries = fao_boundaries.filterBounds(buffered_shapefile.geometry()).filterBounds(buffered_country_boundary.geometry())
        merged_boundary = ee.FeatureCollection(overlapping_boundaries.geometry().dissolve())
        merged_area = merged_boundary.geometry().area().multiply(100).divide(1000000)
        if merged_area.getInfo() < (country_area.getInfo() / 2) and merged_area.getInfo() > 2500000:
            overlapping_boundaries = merged_boundary
            buffered_merged_boundary = merged_boundary.map(lambda feature: feature.buffer(-1000))
            overlapping_l2_boundaries = fao_l2_boundaries.filterBounds(buffered_merged_boundary.geometry())
        else:
            overlapping_boundaries = overlapping_country_boundary
            buffered_country_boundary = overlapping_boundaries.map(lambda feature: feature.buffer(-1000))
            overlapping_l2_boundaries = fao_boundaries.filterBounds(buffered_country_boundary.geometry())
    return overlapping_boundaries, shapefile_ee

# Initialize service account and download JSON key file
service_account = 'jnr-670@jnr-master.iam.gserviceaccount.com'
key_file_name = 'jnr-works/jnr-master.json'
local_key_path = 'jnr-master.json'

# Download the service account JSON key file from S3
download_file_from_s3(key_file_name, local_key_path)

# Initialize Earth Engine
google_cloud_init(service_account, local_key_path)

st.header("Hansen Deforestation Dataset")

row1_col1, row1_col2 = st.columns([6, 1])
width = 1400
height = 800

vis_params = {
    "min": 0,
    "max": 2,
    "palette": ['white', 'orange', 'red', 'green'],
}

Map = geemap.Map(center=[40, 100], zoom=2)
Map.add_colorbar(vis_params, label="Deforestation", layer_name="Deforestation")

years = ["2014", "2019", "2024"]

with row1_col2:
    selected = st.multiselect("Select a year", years)

if selected:
    for year in selected:
        Map.addLayer(get_hansen_data(year), vis_params, year)

    with row1_col1:
        Map.to_streamlit(width=width, height=height)
else:
    with row1_col1:
        Map.to_streamlit(width=width, height=height)
