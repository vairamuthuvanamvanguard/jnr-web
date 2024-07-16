import streamlit as st
import folium
import json
import requests
import geocoder
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from polyline import decode
import pandas as pd
from geopy.distance import geodesic
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

# Function to load GeoJSON data from a file path
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

# Function to get directions from Google Maps API
def get_directions(origin, destination):
    endpoint = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "key": api_key,
        "mode": "driving"
    }
    response = requests.get(endpoint, params=params)
    directions = response.json()
    if directions["status"] == "OK":
        return directions["routes"][0]["overview_polyline"]["points"]
    else:
        st.error("Error fetching directions from Google Maps API.")
        return None

# Streamlit app
st.title("GeoJSON Path and Distance Calculation with Google Maps")

# Load GeoJSON data
geojson_data = load_geojson("output.geojson")

# Extract points from GeoJSON
points = []
for feature in geojson_data['features']:
    coords = feature['geometry']['coordinates']
    points.append((coords[1], coords[0]))

# Get user location using geocoder
g = geocoder.ip('me')
user_location = g.latlng

if user_location:
    lat, lon = user_location
else:
    st.warning("Unable to fetch location. Please check your internet connection.")
    st.stop()

# Create buttons for navigating to the user's location and GeoJSON path
col1, col2 = st.columns([1, 1])
if col1.button("Locate Myself"):
    map_center = user_location
elif col2.button("Locate GeoJSON Path"):
    map_center = points[0] if points else user_location
else:
    map_center = user_location

# Create map centered on chosen location
m = folium.Map(location=map_center, zoom_start=12)

# Cluster nearby points
marker_cluster = MarkerCluster().add_to(m)

# Add points to map
for point in points:
    folium.Marker(
        location=point,
        icon=folium.CustomIcon(icon_image='marker-icon.png', icon_size=(25, 41))
    ).add_to(marker_cluster)

# Add user location marker with custom icon
folium.Marker(
    location=user_location,
    icon=folium.CustomIcon(icon_image='marker-icon.png', icon_size=(25, 41)),
    tooltip="You are here"
).add_to(m)

# Add a progress bar
progress_bar = st.progress(0)
total_points = len(points)

# Initialize a list to store distances
distances = []

# Draw the route from user location to each GeoJSON point
for i, point in enumerate(points):
    directions = get_directions(user_location, point)
    if directions:
        decoded_points = decode(directions)
        folium.PolyLine(
            locations=decoded_points,
            color="green",
            weight=5,
            opacity=0.8
        ).add_to(m)
        # Calculate and store distance
        distance = geodesic(user_location, point).km
        distances.append({
            "Point": point,
            "Distance (km)": distance
        })
    progress_bar.progress((i + 1) / total_points)

# Display the map
st_folium(m, width=1200, height=600)

# Display distances in a table
if distances:
    df = pd.DataFrame(distances)
    st.write("Distances from user location to GeoJSON points:")
    st.dataframe(df)
