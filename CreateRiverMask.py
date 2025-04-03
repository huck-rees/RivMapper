import ee
import geemap
import geopandas as gpd
import pandas as pd
import os
from datetime import datetime
import requests
import zipfile
import io
import math
import ast
import time
ee.Initialize()

import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import shape, Polygon
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt

import glob
import subprocess
import ast
import re

# Normalized Difference Water Index (MNDWI)
def Mndwi(image):
    """
    Calculate the Modified Normalized Difference Water Index (MNDWI) for a given image.

    Parameters:
    image (ee.Image): The input image.

    Returns:
    ee.Image: The resulting image with the MNDWI band named 'mndwi'.
    """
    return image.normalizedDifference(['Green', 'Swir1']).rename('mndwi')

# Modified Bare Soil Reflectance Variables
def Mbsrv(image):
    """
    Calculate the Modified Bare Soil Reflectance Variable (MBSRV) for a given image.

    Parameters:
    image (ee.Image): The input image.

    Returns:
    ee.Image: The resulting image with the MBSRV band named 'mbsrv'.
    """
    return image.select(['Green']).add(image.select(['Red'])).rename('mbsrv')

def Mbsrn(image):
    """
    Calculate the Modified Bare Soil Reflectance Normalized (MBSRN) for a given image.

    Parameters:
    image (ee.Image): The input image.

    Returns:
    ee.Image: The resulting image with the MBSRN band named 'mbsrn'.
    """
    return image.select(['Nir']).add(image.select(['Swir1'])).rename('mbsrn')

# Normalized Difference Vegetation Index (NDVI)
def Ndvi(image):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) for a given image.

    Parameters:
    image (ee.Image): The input image.

    Returns:
    ee.Image: The resulting image with the NDVI band named 'ndvi'.
    """
    return image.normalizedDifference(['Nir', 'Red']).rename('ndvi')

# Automated Water Extraction Index (AWESH)
def Awesh(image):
    """
    Calculate the Automated Water Extraction Index (AWEsh) for a given image.

    Parameters:
    image (ee.Image): The input image with the necessary bands for MBSRN calculation.

    Returns:
    ee.Image: The resulting image with the AWEsh band named 'awesh'.
    """
    return image.expression(
        'Blue + 2.5 * Green + (-1.5) * mbsrn + (-0.25) * Swir2',
        {
            'Blue': image.select(['Blue']),
            'Green': image.select(['Green']),
            'mbsrn': Mbsrn(image).select(['mbsrn']),
            'Swir2': image.select(['Swir2'])
        }
    ).rename('awesh')

# Decision Tree for Surface Water Extent (DSWE)
def Dswe(image):
    """
    Calculate the Decision Tree for Surface Water Extent (DSWE) for a given image.

    Parameters:
    image (ee.Image): The input image with bands required for the DSWE calculation.

    Returns:
    ee.Image: The resulting image with the DSWE classification named 'dswe'.
    """
    mndwi = Mndwi(image)
    mbsrv = Mbsrv(image)
    mbsrn = Mbsrn(image)
    awesh = Awesh(image)
    swir1 = image.select(['Swir1'])
    nir = image.select(['Nir'])
    ndvi = Ndvi(image)
    blue = image.select(['Blue'])
    swir2 = image.select(['Swir2'])

    # Decision tree thresholds
    t1 = mndwi.gt(0.124)
    t2 = mbsrv.gt(mbsrn)
    t3 = awesh.gt(0)
    t4 = (mndwi.gt(-0.44)
          .And(swir1.lt(0.09))
          .And(nir.lt(0.15))
          .And(ndvi.lt(0.7)))
    t5 = (mndwi.gt(-0.5)
          .And(blue.lt(0.1))
          .And(swir1.lt(0.3))
          .And(swir2.lt(0.1))
          .And(nir.lt(0.25)))

    # Combine results using weights to create unique classes
    t = t1.add(t2.multiply(10)).add(t3.multiply(100)).add(t4.multiply(1000)).add(t5.multiply(10000))

    # Define DSWE classification levels
    noWater = t.eq(0).Or(t.eq(1)).Or(t.eq(10)).Or(t.eq(100)).Or(t.eq(1000))
    hWater = t.eq(1111).Or(t.eq(10111)).Or(t.eq(11011)).Or(t.eq(11101)).Or(t.eq(11110)).Or(t.eq(11111))
    mWater = (t.eq(111).Or(t.eq(1011)).Or(t.eq(1101)).Or(t.eq(1110))
              .Or(t.eq(10011)).Or(t.eq(10101)).Or(t.eq(10110))
              .Or(t.eq(11001)).Or(t.eq(11010)).Or(t.eq(11100)))
    pWetland = t.eq(11000)
    lWater = (t.eq(11).Or(t.eq(101)).Or(t.eq(110)).Or(t.eq(1001))
              .Or(t.eq(1010)).Or(t.eq(1100)).Or(t.eq(10000))
              .Or(t.eq(10001)).Or(t.eq(10010)).Or(t.eq(10100)))

    # Assign classification levels to DSWE
    iDswe = (noWater.multiply(0)
             .add(hWater.multiply(1))
             .add(mWater.multiply(2))
             .add(pWetland.multiply(3))
             .add(lWater.multiply(4)))

    return iDswe.rename(['dswe'])

# Generate Binary Water Mask Based on DSWE
def ClassifyWaterJones2019(image, max_level):
    """
    Creates a binary water mask from an input image using DSWE classification.
    
    Args:
        image: The input image containing relevant bands and indices.
        max_level: The maximum classification level (inclusive) to include in the mask.
    
    Returns:
        ee.Image: A single-band image representing the binary water mask.
    """
    dswe = Dswe(image)
    # Start with level 1 (high-confidence water)
    water_mask = dswe.eq(1)

    # Add higher levels if within the specified max level
    for level in range(2, max_level + 1):
        water_mask = water_mask.Or(dswe.eq(level))

    return water_mask.rename(['waterMask'])

def maskL8sr(image):
    """
    Masks out clouds and cloud shadows within Landsat 8 and 9 images using the BQA band.

    Args:
        image: ee.Image, the input Landsat image.

    Returns:
        ee.Image: The cloud-masked image.
    """
    # Bit positions for cloud shadow (bit 3) and clouds (bit 5) in the QA_PIXEL band
    cloud_shadow_bit_mask = (1 << 3)
    clouds_bit_mask = (1 << 5)
    # Select the BQA band
    qa = image.select('BQA')
    # Both bits should be zero for clear conditions
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(
        qa.bitwiseAnd(clouds_bit_mask).eq(0)
    )
    return image.updateMask(mask)

def getLandsatCollection():
    """
    Merges Landsat 5, 7, 8, and 9 collections (Tier 1, Collection 2 SR) 
    and standardizes the band names for consistent analysis.

    Returns:
        ee.ImageCollection: A merged collection of standardized Landsat images.
    """
    # Define the band mappings for each Landsat version
    bn9 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    bn8 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    bn7 = ['SR_B1', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    bn5 = ['SR_B1', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    # Standardized names for all bands
    standard_bands = ['uBlue', 'Blue', 'Green', 'Red', 'Nir', 'Swir1', 'Swir2', 'BQA']

    # Fetch and rename bands in the Landsat collections
    ls5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").select(bn5, standard_bands)
    ls7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterDate('1999-04-15', '2003-05-30').select(bn7, standard_bands)
    ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").select(bn8, standard_bands)
    ls9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").select(bn9, standard_bands)

    # Merge all collections
    merged_collection = ls5.merge(ls7).merge(ls8).merge(ls9)

    return merged_collection

def rescale(image):
    """
    Rescale the reflectance values of Landsat imagery to allow for use of Landsat Collection 2 in DSWE.

    Parameters:
    image (ee.Image): The input image with bands to be rescaled.

    Returns:
    ee.Image: The image with rescaled bands added.
    """
    bns = ['uBlue', 'Blue', 'Green', 'Red', 'Nir', 'Swir1', 'Swir2', 'BQA']
    optical_bands = image.select(bns).multiply(0.0000275).add(-0.2)
    return image.addBands(optical_bands, None, True)

def load_watershed_shapefile(folder_name, base_path):
    """
    Loads a watershed shapefile or reach shapefile into an Earth Engine FeatureCollection.

    Parameters:
        - folder_name: str, the name of the specific watershed/reach folder/shapefile. This can include one or more features.
        - base_path: str, optional base directory containing watershed folders (default set to your path).

    Returns:
        - ee.FeatureCollection: An Earth Engine FeatureCollection representing the watershed geometry.
    """
    # Construct the full path to the shapefile
    shapefile_path = os.path.join(base_path, folder_name, f"{folder_name}.shp")
    
    # Check if the shapefile exists
    if not os.path.exists(shapefile_path):
        print(f"No shapefile found at {shapefile_path}")
        return None
    
    # Load the shapefile into a GeoDataFrame
    try:
        gdf = gpd.read_file(shapefile_path)
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None
    
    # Initialize the Earth Engine API if not already done
    if not ee.data._credentials:
        ee.Initialize()
    
    # Convert the GeoDataFrame to a GeoJSON-like structure and then to a FeatureCollection
    try:
        feature_collection = geemap.geopandas_to_ee(gdf)
    except Exception as e:
        print(f"Error converting GeoDataFrame to FeatureCollection: {e}")
        return None

    return feature_collection

# Function to generate a water mask for a given year and polygon feature
def get_water_mask_for_feature(year, max_level, polygon):
    """
    Generate a water mask for a given year and polygon feature using Landsat imagery.

    Parameters:
    year (int): The year for which to generate the water mask.
    max_level (float): The maximum DSWE water level threshold for classification.
    polygon (ee.Geometry.Polygon): The polygon feature defining the area of interest.

    Returns:
    ee.Image: The water mask image for the specified year and polygon.
    """
    imagery = (getLandsatCollection()
               .map(maskL8sr)
               .map(rescale)
               .filterDate(f'{year}-01-01', f'{year}-12-31')
               .filterBounds(polygon))

    image_composite = imagery.median().clip(polygon)
    water_mask = ClassifyWaterJones2019(image_composite, max_level)

    return water_mask

# Main function to generate water masks for each feature in a FeatureCollection
def get_water_masks(year, folder_name, max_level, base_path=r'C:\Users\huckr\Desktop\UCSB\Dissertation\Data\RiverMapping\Reaches'):
    """
    Generate water masks for each feature in a FeatureCollection for a given year.

    Parameters:
    year (int): The year for which to generate the water masks.
    folder_name (str): The name of the folder containing the study area shapefile.
    max_level (float): The maximum DSWE water level threshold for classification.
    base_path (str): The base path to the directory containing the study area shapefile.

    Returns:
    ee.FeatureCollection: A FeatureCollection with water masks for each feature.
    """
    
    # Load study area shapefile, using the base path
    polygon_fc = load_watershed_shapefile(folder_name, base_path)

    # Map over each feature in the FeatureCollection and generate water masks
    water_masks = polygon_fc.map(lambda feature: get_water_mask_for_feature(year, max_level, feature.geometry()).set('polygon_id', feature.id()))

    return water_masks

# Load river reach shapefiles and merge them
def load_and_merge_reaches(csv_path):
    # Load the CSV file
    river_data = pd.read_csv(csv_path)
    
    # List to hold individual reach GeoDataFrames
    gdfs = []
    
    # Iterate through each row of the CSV
    for _, row in river_data.iterrows():
        # Build the full path to the reach shapefile
        reach_base_path = row['output_reaches_path']
        river_name = row['river_name']
        shapefile_path = os.path.join(reach_base_path, river_name, f"{river_name}.shp")
        
        # Load the reach shapefile as a GeoDataFrame
        gdf = gpd.read_file(shapefile_path)
        gdfs.append(gdf)
    
    # Merge all individual GeoDataFrames into a single GeoDataFrame
    reach_merge = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    return reach_merge, river_data['grwl_directory'].iloc[0]

# Convert Google Drive link to direct download link
def convert_drive_to_download_url(drive_url):
    file_id = drive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return download_url

# Download and extract shapefiles from Google Drive
def download_and_extract_shapefiles(download_url, extract_folder="shapefile_data"):
    response = requests.get(download_url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_folder)
            extracted_files = z.namelist()
            print(f"Extracted files: {extracted_files}")
        return extract_folder
    else:
        raise Exception(f"Failed to download the file. Status code: {response.status_code}")

# Load shapefile from a specified directory
def load_shapefile(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".shp"):
                shapefile_path = os.path.join(root, file)
                gdf = gpd.read_file(shapefile_path)
                return gdf
    raise Exception("No shapefile found in the specified directory.")

# Reproject 'tiles' to match 'reaches' CRS and find intersections
def reproject_and_intersect(tiles_gdf, reaches):
    tiles_reprojected = tiles_gdf.to_crs(reaches.crs)
    intersection = gpd.overlay(reaches, tiles_reprojected, how='intersection')
    return intersection['TILE_ID'].unique().tolist()

# Load and merge matching shapefiles from a directory based on 'tile_ids'
def load_matching_shapefiles(directory, tile_ids):
    gdfs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            for tile_id in tile_ids:
                if file.startswith(tile_id) and file.endswith(".shp"):
                    shapefile_path = os.path.join(root, file)
                    gdf = gpd.read_file(shapefile_path)
                    gdfs.append(gdf)
                    print(f"Loaded shapefile: {file}")
    
    if not gdfs:
        raise Exception("No matching shapefiles found.")
    
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return merged_gdf

# gets final grwl subset
def grwl_main(csv_path, drive_url):
    # Step 1: Load and merge river reach shapefiles
    reach_merge, grwl_directory = load_and_merge_reaches(csv_path)
    
    # Step 2: Convert Drive URL to direct download
    download_url = convert_drive_to_download_url(drive_url)
    
    # Step 3: Download and extract the tiles shapefiles
    extracted_folder = download_and_extract_shapefiles(download_url)
    
    # Step 4: Load the extracted shapefile as GeoDataFrame
    tiles_gdf = load_shapefile(extracted_folder)
    
    # Step 5: Find intersecting 'TILE_ID' values between 'tiles' and 'reaches'
    tile_ids = reproject_and_intersect(tiles_gdf, reach_merge)
    
    # Step 6: Load and merge matching shapefiles from GRWL data
    grwl_gdf = load_matching_shapefiles(grwl_directory, tile_ids)
    
    # Return the final merged GeoDataFrame
    return grwl_gdf

def get_utm_epsg(longitude, latitude):
    """
    Determine the correct UTM EPSG code based on the centroid's longitude and latitude.
    """
    utm_zone = int((longitude + 180) / 6) + 1
    if latitude >= 0:
        epsg_code = 32600 + utm_zone  # UTM zones for northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # UTM zones for southern hemisphere
    return epsg_code

def raster_to_gdf(labeled_array, transform, crs=None):
    """
    Convert a labeled raster array to a GeoDataFrame with polygons for each labeled region.
    """
    polygons, labels = [], []
    for geom, value in rasterio.features.shapes(labeled_array, transform=transform):
        if value != 0:  # Skip background (value 0)
            polygons.append(shape(geom))  # Convert to Shapely geometry
            labels.append(value)

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'label': labels})
    if crs is not None:
        gdf = gdf.set_crs(crs)
    return gdf

def find_intersecting_pixels(raster_gdf, buff_gdf):
    """
    Find unique pixel values that the buffer intersects with.
    """
    intersections = gpd.sjoin(raster_gdf, buff_gdf, how='inner', predicate='intersects')
    unique_labels = set(intersections['label'])
    return unique_labels

def filter_and_reclassify_raster(classified_raster, unique_labels, min_pixels=200):
    """
    Filter the classified raster to include only specified components and reclassify pixels.
    Eliminate components smaller than min_pixels.
    """
    final_raster = np.zeros_like(classified_raster, dtype=np.int32)
    for unique_label in unique_labels:
        final_raster[classified_raster == unique_label] = 1

    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(final_raster, structure=structure)

    component_sizes = np.bincount(labeled_array.ravel())
    for component_label, size in enumerate(component_sizes):
        if size < min_pixels:
            final_raster[labeled_array == component_label] = 0

    return final_raster

def export_raster(raster, transform, crs, output_path):
    """
    Export a raster array to a GeoTIFF file with LZW compression.
    
    Parameters:
        raster (numpy array): The raster data to export.
        transform (Affine): Affine transform for the raster.
        crs (CRS): Coordinate reference system of the raster.
        output_path (str): Path to save the output GeoTIFF file.
    """
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,  # Assuming single-band raster
        dtype=raster.dtype,
        crs=crs,
        transform=transform,
        compress='LZW'  # Apply LZW compression to reduce file size
    ) as dst:
        dst.write(raster, 1)

    print(f"Raster successfully saved with compression to: {output_path}")

def cleaner_main(raster_path, grwl, reach, min_pixels, output_path):
    """
    The main function that orchestrates the cleaning of the raster.
    """
    # Load raster
    with rasterio.open(raster_path) as src:
        water_mask = src.read(1)  # Read the first band (water mask)
        transform = src.transform  # Get the raster's affine transform
        crs = src.crs  # Get the raster's CRS
    
    reach = reach.to_crs(grwl.crs)
    
    # Clip and dissolve
    grwl_reach = gpd.clip(grwl, reach)
    grwl_reach_dissolved = grwl_reach.dissolve()

    # Get UTM zone for projection
    centroid = grwl_reach_dissolved.geometry.centroid.iloc[0]
    longitude, latitude = centroid.x, centroid.y
    utm_epsg = get_utm_epsg(longitude, latitude)
    grwl_reach_projected = grwl_reach_dissolved.to_crs(epsg=utm_epsg)

    # Buffer the projected GRWL
    mean_width = grwl_reach['width_m'].mean()
    grwl_buff = grwl_reach_projected.buffer(mean_width)

    # Label the connected components in the water mask
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled_array, num_features = label(water_mask, structure=structure)

    # Convert the labeled array to a GeoDataFrame
    mask_gdf = raster_to_gdf(labeled_array, transform, crs=crs)
    mask_gdf = mask_gdf.to_crs(grwl_buff.crs)

    # Convert the buffer to a GeoDataFrame
    grwl_buff_gdf = gpd.GeoDataFrame(geometry=grwl_buff, crs=grwl_buff.crs)

    # Find the intersecting pixels
    unique_pix = find_intersecting_pixels(mask_gdf, grwl_buff_gdf)

    # Filter and reclassify the raster
    final_raster = filter_and_reclassify_raster(labeled_array, unique_pix, min_pixels=min_pixels)

    # Export the cleaned raster
    export_raster(final_raster, transform, crs, output_path)

    print(f"Processes raster saved to: {output_path}")

# Function to export masks for a range of years and features within a polygon feature collection
def export_masks(folder_name, max_level, grwl_gdf, year_range="All", reach_range="All",
                 base_path=r'C:\Users\huckr\Desktop\UCSB\Dissertation\Data\RiverMapping\Reaches', 
                 local_output_path=r'C:\Users\huckr\Desktop\UCSB\Dissertation\Data\RiverMapping\RiverMasks',
                 min_pix=0, 
                 min_tile_size_km2=400):
    """
    Generates water masks for a given river and specified range of years and reaches.
    If export fails due to size, the image is tiled and exported in smaller parts,
    and the tiles are stitched back into a single .tif file.

    Parameters:
        year_range (str, int, tuple, optional): Range of years (start, end), a single year, or "All".
        reach_range (str, int, tuple, optional): Range of reaches (start, end), a single reach, or "All".
        folder_name (str): Name of the folder containing watershed shapefile.
        max_level (int): Maximum DSWE level for water mask generation.
        base_path (str, optional): Base path to the folder containing watershed shapefiles.
        local_output_path (str, optional): Local path where the output masks will be saved.
        min_pix (int, optional): Minimum pixel count for post-processing.
        min_tile_size_km2 (float, optional): Minimum tile size in square kilometers.

    Outputs:
        Exports water masks as TIFF files to the specified local output path.
    """

    def get_geometry_area_km2(geom):
        """Calculate the area of a given ee.Geometry in square kilometers."""
        return geom.area().divide(1e6).getInfo()  # Convert from m² to km²

    def tile_geometry(geometry, tile_size_km2):
        """Tiles the geometry into smaller parts based on the target tile size."""
        area_km2 = get_geometry_area_km2(geometry)
        if area_km2 <= tile_size_km2:
            return [geometry]  # No need to tile, small enough

        # Determine how many tiles we need based on the area
        num_tiles = math.ceil(area_km2 / tile_size_km2)
        tile_count_per_side = math.ceil(math.sqrt(num_tiles))  # Number of tiles per side

        # Get the bounding box and tile the region
        bounds = geometry.bounds()
        bbox_coords = bounds.coordinates().getInfo()[0]
        min_lon, min_lat = bbox_coords[0]
        max_lon, max_lat = bbox_coords[2]

        lon_step = (max_lon - min_lon) / tile_count_per_side
        lat_step = (max_lat - min_lat) / tile_count_per_side

        tiles = []
        for i in range(tile_count_per_side):
            for j in range(tile_count_per_side):
                tile = ee.Geometry.Rectangle([
                    [min_lon + i * lon_step, min_lat + j * lat_step],
                    [min_lon + (i + 1) * lon_step, min_lat + (j + 1) * lat_step]
                ])
                tiles.append(tile)
        return tiles

    def export_tile(tile_geom, output_file_path, year, feature_id):
        """Attempts to export a single tile, retrying if necessary."""
        for retry in range(1):  # Retry one time for network or transient errors
            try:
                water_mask = get_water_mask_for_feature(year, max_level, tile_geom)
                water_mask_reprojected = water_mask.reproject(crs=target_crs, scale=30)

                geemap.ee_export_image(
                    water_mask_reprojected,
                    filename=output_file_path,
                    scale=30,
                    region=tile_geom.getInfo()['coordinates'],
                    file_per_band=False
                )
                
                # Wait for the export to complete and check if file exists
                time.sleep(10)
                if os.path.exists(output_file_path):
                    print(f"RAW: Successfully exported {output_file_path}.")
                    return True
                else:
                    print(f"Export failed: File {output_file_path} not found.")
            except Exception as e:
                error_message = str(e)
                print(f"Failed to export {output_file_path} on retry {retry + 1}. Error: {error_message}")
                time.sleep(5)  # Delay before retrying
                
                # Check if the error is related to missing bands or invalid image data
                if "Image.select" in error_message or "no bands" in error_message:
                    print(f"Skipping tiling for {year}, reach {feature_id} due to image generation error.")
                    return "invalid_image"  # Return specific error message if image is invalid
        return "size_limit"  # Return specific message if it fails due to size

    def stitch_tiles(output_folder, reach_id, year, output_file_name):
        """Stitches multiple tiles back into a single .tif using Rasterio, applies compression, and deletes the tiles after successful stitching."""
        # Find all .tif tiles in the output folder that match the reach, year, and include 'tile' in the file name
        tile_files = glob.glob(os.path.join(output_folder, f"*reach_{reach_id}_{year}_tile_*.tif"))

        if len(tile_files) > 1:
            # Open the tile files and merge them using Rasterio
            tile_datasets = [rasterio.open(tile) for tile in tile_files]
            mosaic, out_transform = merge(tile_datasets)

            # Define the output path for the stitched file
            merged_output_path = os.path.join(output_folder, output_file_name)
            print(f"Stitching tiles into {merged_output_path}...")

            # Copy the metadata from one of the tiles (all tiles should have similar metadata)
            out_meta = tile_datasets[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "LZW"  # Apply LZW compression
            })

            # Write the mosaic to a new .tif file with compression
            with rasterio.open(merged_output_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            # Close all tile datasets
            for ds in tile_datasets:
                ds.close()

            print(f"Tiles successfully stitched into {merged_output_path}.")

            # Delete the tiles after successful stitching
            for tile in tile_files:
                try:
                    os.remove(tile)
                    print(f"Deleted tile: {tile}")
                except OSError as e:
                    print(f"Error deleting tile {tile}: {e}")

        elif len(tile_files) == 1:
            print("Only one tile found, no stitching needed.")
        else:
            print("No tiles found for stitching.")

    # Load watershed shapefile
    polygon_fc = load_watershed_shapefile(folder_name, base_path)

    # Define the target CRS
    target_crs = 'EPSG:4326'

    # Determine the year range
    if year_range == "All":
        year_start = 1984
        year_end = 2025
    elif isinstance(year_range, int):
        year_start = year_range
        year_end = year_range
    elif isinstance(year_range, str):
        # Check if the string is in the format "(YYYY, YYYY)"
        if re.match(r'^\(\d{4}, \d{4}\)$', year_range):  # Match the pattern (YYYY, YYYY)
            try:
                # Convert the string to a tuple of integers
                year_range = ast.literal_eval(year_range)
                year_start, year_end = year_range
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid year range format: {year_range}")
        else:
            raise ValueError(f"Invalid string format for year_range: {year_range}")
    elif isinstance(year_range, tuple) and len(year_range) == 2:
        year_start, year_end = year_range
    else:
        raise ValueError("year_range must be 'All', an int, or a tuple (start, end).")

    # Determine the reach range
    if reach_range == "All":
        reach_start = 1
        reach_end = 9999
    elif isinstance(reach_range, int):
        reach_start = reach_range
        reach_end = reach_range
    elif isinstance(reach_range, str):
        # Check if the string is in the format "(XX, YY)"
        if re.match(r'^\(\d{1,4}, \d{1,4}\)$', reach_range):  # Match (XX, YY) with 1 to 4 digits
            try:
                # Convert the string to a tuple of integers
                reach_range = ast.literal_eval(reach_range)
                reach_start, reach_end = reach_range
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid reach range format: {reach_range}")
        else:
            raise ValueError(f"Invalid string format for reach_range: {reach_range}")
    elif isinstance(reach_range, tuple) and len(reach_range) == 2:
        reach_start, reach_end = reach_range
    else:
        raise ValueError("reach_range must be 'All', an int, or a tuple (start, end).")

    # Loop through each feature in the FeatureCollection
    for feature in polygon_fc.getInfo()['features']:

        # Get the geometry and ID of the feature
        feature_geom = ee.Feature(feature).geometry()
        feature_id = int(feature['properties']['ds_order'])

        # Check if the feature_id is within the reach range
        if reach_start <= feature_id <= reach_end:
            # Create the reach subfolder if it does not exist
            reach_folder_path = os.path.join(local_output_path, folder_name, f"reach_{feature_id}", "Raw")
            if not os.path.exists(reach_folder_path):
                os.makedirs(reach_folder_path)

            for year in range(year_start, year_end + 1):
                # Set the output file path based on the folder, year, and feature ID
                output_file_name = f"{folder_name}_reach_{feature_id}_{year}_DSWE_level_{max_level}.tif"
                output_file_path = os.path.join(reach_folder_path, output_file_name)

                # Attempt to export the full water mask
                try:
                    export_status = export_tile(feature_geom, output_file_path, year, feature_id)

                    if export_status == "invalid_image":
                        print(f"Skipping further processing for {year}, reach {feature_id} due to invalid image.")
                        continue  # Skip to the next feature/year if invalid image

                    elif export_status == "size_limit":
                        # If the export failed due to size, tile the geometry and retry
                        print(f"Export failed for {output_file_name}, attempting tiling...")
                        tiles = tile_geometry(feature_geom, min_tile_size_km2)

                        tile_paths = []
                        for tile_idx, tile in enumerate(tiles):
                            tile_output_name = f"{folder_name}_reach_{feature_id}_{year}_tile_{tile_idx}_DSWE_level_{max_level}.tif"
                            tile_output_path = os.path.join(reach_folder_path, tile_output_name)
                            tile_paths.append(tile_output_path)
                            if not export_tile(tile, tile_output_path, year, feature_id):
                                print(f"Failed to export tile {tile_idx} for {output_file_name}.")
                        
                        # After tiling, stitch the tiles back together
                        stitch_tiles(reach_folder_path, feature_id, year, output_file_name)
                    
                    # Only proceed if the export was successful
                    process_folder_path = os.path.join(local_output_path, folder_name, f"reach_{feature_id}", "Processed")
                    if not os.path.exists(process_folder_path):
                        os.makedirs(process_folder_path)
        
                    raster_name = f"{folder_name}_reach_{feature_id}_{year}_DSWE_level_{max_level}.tif"
                    output_path = os.path.join(process_folder_path, output_file_name)
        
                    reach_name = f'{folder_name}.shp'
                    reaches = gpd.read_file(os.path.join(base_path, folder_name, reach_name))
                    reach = reaches[reaches['ds_order'] == feature_id]
        
                    cleaner_main(output_file_path, grwl_gdf, reach, min_pix, output_path)
                    print(f"PROCESSED: Exported {raster_name} to {process_folder_path}.")

                except Exception as e:
                    print(f"Failed to export {output_file_name}. Error: {e}")
                    continue  # Skip to the next iteration

def process_multiple_rivers(csv_file):
    
    drive_url = "https://drive.google.com/file/d/1K6x1E0mmLc0k7er4NCIeaZsTfHi2wxxI/view?usp=sharing"
    # Read the CSV file containing input variables for multiple rivers
    river_data = pd.read_csv(csv_file)
    grwl_gdf = grwl_main(csv_file, drive_url)

    # Loop through each row (each river)
    for index, row in river_data.iterrows():
        # Extract the necessary input values from the CSV row
        Folder_name = row['river_name']
        DSWE_threshold_level = row['DSWE_threshold_level']
        year_range = row['year_range']
        reach_range = row['reach_range']
        base_path = row['output_reaches_path']
        local_output_path = row['mask_output_path']
        min_pix = row['min_pixels']
        
        # Call the function to process the river mask for the current river
        export_masks(Folder_name, 
                     DSWE_threshold_level,
                     grwl_gdf,
                     year_range, 
                     reach_range, 
                     base_path, 
                     local_output_path,
                     min_pix)
        
csv_file_path = r"/home/jamesrees/Desktop/RivMapper/Data/RiverMapping/Bermejo_river_datasheet.csv" # Replace with the actual path to your CSV
process_multiple_rivers(csv_file_path)