import streamlit as st
import geemap.foliumap as geemap
import leafmap.foliumap as leafmap
import geopandas as gpd
from utils import download_file_from_s3, upload_file_to_s3, google_cloud_init
from udef import UDefAllocation
import os

st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
S3_REGION = os.getenv('S3_REGION')

st.title('Unplanned Deforestation Allocation using Benchmark Approach')

# Ensure the path to your service account JSON key file is correct
service_account = 'jnr-670@jnr-master.iam.gserviceaccount.com'
key_file_name = 'jnr-works/jnr-master.json'
local_key_path = 'jnr-master.json'

# Download the service account JSON key file from S3
download_file_from_s3(key_file_name, local_key_path)

# Initialize Earth Engine
google_cloud_init(service_account, local_key_path)

chunksize = 3000

# Ensure local directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('jnr-works/out', exist_ok=True)

col1, col2, col3 = st.columns([2, 2, 2])

def download_shapefile_components(s3_path, local_dir):
    """Download all components of a shapefile from S3."""
    base_name = os.path.splitext(s3_path)[0]
    for ext in [".shp", ".shx", ".dbf", ".prj"]:
        s3_file = base_name + ext
        local_file = os.path.join(local_dir, os.path.basename(s3_file))
        download_file_from_s3(s3_file, local_file)
    return local_dir

with col1:
    fcc = st.text_input(
        "Enter path to the FCC raster",
        'jnr-works/data/lct_vanam_palsarl8_1566v4.tif',
    )
    fcc_local_path = os.path.basename(fcc)
    download_file_from_s3(fcc, fcc_local_path)
    
    outpath = st.text_input(
        "Specify Output directory",
        'jnr-works/out/',
    )

with col2:
    jurisdiction = st.text_input(
        "Enter output to jurisdiction shapefile",
        "jnr-works/data/vichada_boundary1566.shp",
    )
    jurisdiction_local_dir = "data"
    download_shapefile_components(jurisdiction, jurisdiction_local_dir)
    
    district = st.text_input(
        "Enter output to l2 Boundary",
        "jnr-works/data/vichada_l2_boundary1566.shp",
    )
    district_local_dir = "data"
    download_shapefile_components(district, district_local_dir)

with col3:
    l2_raster_name = st.text_input(
        "Enter the name of l2 raster",
        "l2_raster.tif",
    )
    
    activity_data = st.text_input(
        "Enter activity data",
        "50000",
    )

fr = UDefAllocation(outpath='.')
fr.load_data(
    fcc=fcc_local_path,
    jurisdiction=os.path.join(jurisdiction_local_dir, "vichada_boundary1566.shp"),
    next_admin_boundary=os.path.join(district_local_dir, "vichada_l2_boundary1566.shp"),
    chunksize=chunksize
)
fr.create_l2_raster(l2_raster_name=l2_raster_name)

summary = fr.get_nrt(show=False)

# Upload distance_stats.png to S3
distance_stats_path = 'distance_stats.png'
if os.path.exists(distance_stats_path):
    upload_file_to_s3(distance_stats_path, os.path.join(outpath, distance_stats_path))
    st.image(distance_stats_path)
else:
    st.error(f"{distance_stats_path} does not exist.")

#----------------------------------------------------------------------------------------------------------------
col4, col5 = st.columns([2, 2])

rel_freq1, fit_density1 = fr.fit(stage='testing')
test_fit_density_path = 'test_fit_density.tif'
fr.write(fit_density1, filename=test_fit_density_path)
upload_file_to_s3(test_fit_density_path, os.path.join(outpath, test_fit_density_path))

pred_rel_freq1, adjusted_density1 = fr.predict(stage='testing', max_iter=100)
test_adjusted_density_path = 'test_adjusted_density.tif'
fr.write(adjusted_density1, filename=test_adjusted_density_path)
upload_file_to_s3(test_adjusted_density_path, os.path.join(outpath, test_adjusted_density_path))

test_pred_density_path = 'test_pred_density.png'
fr.plot(adjusted_density1, outfile=test_pred_density_path, cmap=fr.colormap(ctype=2), figsize=(4, 3))
upload_file_to_s3(test_pred_density_path, os.path.join(outpath, test_pred_density_path))

if os.path.exists(test_pred_density_path):
    with col4:
        st.markdown("<h1 style='text-align: center; color: blue;'>Testing Phase Prediction</h1>", unsafe_allow_html=True)
        st.image(test_pred_density_path)
else:
    st.error(f"{test_pred_density_path} does not exist.")

rel_freq2, fit_density2 = fr.fit(stage='application')
app_fit_density_path = 'app_fit_density.tif'
fr.write(fit_density2, filename=app_fit_density_path)
upload_file_to_s3(app_fit_density_path, os.path.join(outpath, app_fit_density_path))

pred_rel_freq2, adjusted_density2 = fr.predict(stage='application', activity_data=float(activity_data), max_iter=100)
app_adjusted_density_path = 'app_adjusted_density.tif'
fr.write(adjusted_density2, filename=app_adjusted_density_path)
upload_file_to_s3(app_adjusted_density_path, os.path.join(outpath, app_adjusted_density_path))

app_pred_density_path = 'app_pred_density.png'
fr.plot(adjusted_density2, outfile=app_pred_density_path, cmap=fr.colormap(ctype=2), figsize=(4, 3))
upload_file_to_s3(app_pred_density_path, os.path.join(outpath, app_pred_density_path))

if os.path.exists(app_pred_density_path):
    with col5:
        st.markdown("<h1 style='text-align: center; color: blue;'>Application Phase Prediction</h1>", unsafe_allow_html=True)
        st.image(app_pred_density_path)
else:
    st.error(f"{app_pred_density_path} does not exist.")
