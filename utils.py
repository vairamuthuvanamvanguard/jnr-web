import boto3
import ee
import os
import streamlit as st
from botocore.exceptions import NoCredentialsError
import tempfile
import uuid
# Load environment variables from .env file
load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
S3_REGION = os.getenv('S3_REGION')

def aws_init():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION
    )
    return session.client('s3')

def download_file_from_s3(file_name, local_path):
    s3 = aws_init()
    try:
        s3.download_file(BUCKET_NAME, file_name, local_path)
        st.success(f"Successfully downloaded {file_name} from S3")
    except NoCredentialsError:
        st.error("Credentials not available")
    except Exception as e:
        st.error(f"Error downloading {file_name}: {str(e)}")

def upload_file_to_s3(local_path, file_name):
    s3 = aws_init()
    try:
        s3.upload_file(local_path, BUCKET_NAME, file_name)
        st.success(f"Successfully uploaded {file_name} to S3")
    except NoCredentialsError:
        st.error("Credentials not available")
    except Exception as e:
        st.error(f"Error uploading {file_name}: {str(e)}")

def google_cloud_init(service_account, key_path):
    try:
        credentials = ee.ServiceAccountCredentials(service_account, key_path)
        ee.Initialize(credentials)
        print("Google Cloud initialized")
    except ee.EEException as e:
        st.error(f"Google Cloud initialization failed: {str(e)}")

def save_uploaded_file(file_content, file_name):
    """
    Save the uploaded file to a temporary directory
    """
    _, file_extension = os.path.splitext(file_name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as f:
        f.write(file_content.getbuffer())
    return file_path

def get_filename(data, url):
    if data or url:
        if data:
            file_path = save_uploaded_file(data, data.name)
            layer_name = os.path.splitext(data.name)[0]
            return file_path, layer_name
        elif url:
            file_path = url
            layer_name = url.split("/")[-1].split(".")[0]
            return file_path, layer_name
    else:
        raise ValueError('Give a valid file path')
