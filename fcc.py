import ee
import os

import geemap.foliumap as geemap
import leafmap.foliumap as leafmap
import geopandas as gpd
from utils import download_file_from_s3, upload_file_to_s3, google_cloud_init, get_filename

service_account = 'jnr-670@jnr-master.iam.gserviceaccount.com'
key_path = '/home/vmmuthu31/jnr-master.json'

key_file_name = 'jnr-works/jnr-master.json'
local_key_path = 'jnr-master.json'

# Download the service account JSON key file from S3
download_file_from_s3(key_file_name, local_key_path)

# Initialize Earth Engine
service_account = 'jnr-670@jnr-master.iam.gserviceaccount.com'
google_cloud_init(service_account, local_key_path)

def categorize_changes(image1, image2, image3):
	# Scenario 1: Change from 1 in image1 to 2 in image2 and image3
	condition1 = image1.eq(1).And(image2.eq(0)).And(image3.eq(0))

	# Scenario 2: Change from 1 in image1, 1 in image2 to 2 in image3
	condition2 = image1.eq(1).And(image2.eq(1)).And(image3.eq(0))

	# Scenario 3: Remains 1 in all three images
	condition3 = image1.eq(1).And(image2.eq(1)).And(image3.eq(1))

	# Create a new raster based on the conditions
	new_image = ee.Image(0).where(condition1, 1) \
		          .where(condition2, 2) \
		          .where(condition3, 3)

	# Value for all other pixels would be 0
	new_image = new_image.where(new_image.eq(0), 0)
	return new_image
	
def get_hansen_data(year):
        if year=="2014":
             nyear = 13
        elif year=="2019":
             nyear = 18
        else:
             nyear = 22

        # Import the NLCD collection.
        gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

        # Tree cover, loss, and gain
        treecover = gfc.select(["treecover2000"])
        lossyear = gfc.select(["lossyear"])

        # Forest in 2000
        forest2000 = treecover.gte(10)
        #forest2000 = forest2000.toByte()

        # Deforestation
        loss = lossyear.gte(1).And(lossyear.lte(nyear))

        # Forest
        forest = forest2000.where(loss.eq(1), 0)
        return forest

def getdata(boundry):
	# Import the NLCD collection.
	gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").clip(boundry)

	# Tree cover, loss, and gain
	treecover = gfc.select(["treecover2000"])
	lossyear = gfc.select(["lossyear"])

	# Forest in 2000
	forest2000 = treecover.gte(10)
	#forest2000 = forest2000.toByte()

	# Deforestation
	loss1 = lossyear.gte(1).And(lossyear.lte(13))
	loss2 = lossyear.gte(1).And(lossyear.lte(18))
	loss3 = lossyear.gte(1).And(lossyear.lte(23))

	# Forest
	forest2014 = forest2000.where(loss1.eq(1), 0)
	forest2019 = forest2000.where(loss2.eq(1), 0)
	forest2024 = forest2000.where(loss3.eq(1), 0)

	change_raster_hansen = categorize_changes(forest2014, forest2019, forest2024)
	return change_raster_hansen
    
def check_and_reproject(shapefile_path, target_crs="EPSG:4326"):
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
        reprojected_path = shapefile_path.replace(".shp", "_reprojected.shp")
        gdf.to_file(reprojected_path)
        return reprojected_path
    return shapefile_path


def get_fao_data(url):
    url = check_and_reproject(url)  # Ensure the shapefile is in EPSG:4326
    shapefile_ee = geemap.shp_to_ee(url)
    user_shapefile = ee.FeatureCollection(shapefile_ee)
    
    try:
        buffered_shapefile = user_shapefile.map(lambda feature: feature.buffer(-1000))

        fao_country_boundary = ee.FeatureCollection('FAO/GAUL/2015/level0')
        fao_boundaries = ee.FeatureCollection('FAO/GAUL/2015/level1')
        fao_l2_boundaries = ee.FeatureCollection('FAO/GAUL/2015/level2')

        overlapping_country_boundary = fao_country_boundary.filterBounds(buffered_shapefile.geometry())
        overlapping_boundaries = fao_boundaries.filterBounds(buffered_shapefile.geometry())
        buffered_l2_shapefile = overlapping_boundaries.map(lambda feature: feature.buffer(-1000))
        overlapping_l2_boundaries = fao_l2_boundaries.filterBounds(buffered_l2_shapefile.geometry())

        merged_boundary = ee.FeatureCollection(overlapping_boundaries.geometry().dissolve())
        merged_area = merged_boundary.geometry().area().multiply(100).divide(1000000)  # Convert to hectares
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
    except Exception as e:
        st.error(f"Error processing FAO data: {str(e)}")
        return None, None
