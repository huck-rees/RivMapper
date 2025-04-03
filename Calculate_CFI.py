import os
import re
import logging
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd
from fractions import Fraction

import rasterio
from rasterio.plot import show
from rasterio.transform import xy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from skimage.morphology import skeletonize, label
from skimage.measure import regionprops
from skimage import io, img_as_bool
from skimage.feature import corner_harris, corner_peaks

from shapefile import Reader, Writer
from shapely.geometry import LineString, Point, MultiLineString, MultiPoint
from shapely.ops import split, linemerge, snap, nearest_points, unary_union
from shapely.strtree import STRtree

from scipy.ndimage import label as scipy_label, find_objects
from scipy.spatial import cKDTree

from rtree import index as rtree_index

import networkx as nx

from collections import Counter, defaultdict

# Function to load raster data
def load_raster(file_path):
    """
    Loads a raster file and returns the data of the first band along with its metadata.

    Parameters:
    file_path (str): The path to the raster file.

    Returns:
    tuple: A tuple containing:
        - data (numpy.ndarray): The data of the first band of the raster.
        - metadata (dict): The metadata of the raster file.
    """
    with rasterio.open(file_path) as dataset:
        data = dataset.read(1)  # Read the first band
        metadata = dataset.meta
    return data, metadata

# Function to save a raster file
def save_raster(output_path, data, metadata):
    """
    Saves a raster file with the given data and metadata.

    Parameters:
    output_path (str): The path to save the output raster file.
    data (numpy.ndarray): The data to be written to the raster file.
    metadata (dict): The metadata of the raster file, including CRS and transform information.

    Returns:
    None
    """
    with rasterio.open(
        output_path, 
        'w', 
        driver='GTiff', 
        height=data.shape[0], 
        width=data.shape[1], 
        count=1, 
        dtype='uint8', 
        crs=metadata['crs'], 
        transform=metadata['transform']
    ) as dst:
        dst.write(data.astype('uint8'), 1)

def eliminate_small_islands(water_mask, min_size=10):
    # Label connected components in the inverse water mask (0 becomes 1, 1 becomes 0)
    labeled_array, num_features = scipy_label(1 - water_mask)
    
    # Iterate through each labeled component
    for i in range(1, num_features + 1):
        blob = labeled_array == i
        if np.sum(blob) <= min_size:
            water_mask[blob] = 1  # Change small no-water blobs to water (1)

    # Now label connected components in the original water mask to find small water blobs
    labeled_array, num_features = scipy_label(water_mask)
    
    # Iterate through each labeled component
    for i in range(1, num_features + 1):
        blob = labeled_array == i
        if np.sum(blob) <= min_size:
            water_mask[blob] = 0  # Change small water blobs to no-water (0)
    
    return water_mask

# Function to perform conditional dilation
def conditional_dilation(image, radius=5):
    """
    Performs a conditional dilation on a binary image. Pixels with a value of 1 that have 
    two or fewer neighbors with the same value will cause a dilation within a given radius.

    Parameters:
    image (numpy.ndarray): The input binary image (2D array) to be processed.
    radius (int, optional): The radius for the dilation operation. Default is 5.

    Returns:
    numpy.ndarray: The dilated image.
    """
    dilated_image = np.copy(image)
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            if image[row, col] == 1:
                neighbors = image[row-1:row+2, col-1:col+2]
                if np.sum(neighbors) <= 2:  # Include the pixel itself in the count
                    dilated_image[max(0, row-radius):min(row+radius+1, image.shape[0]), 
                                  max(0, col-radius):min(col+radius+1, image.shape[1])] = 1
    return dilated_image

# Function to keep only the largest connected component
def keep_largest_component(image):
    """
    Identifies and retains the largest connected component in a binary image. All other components are removed.

    Parameters:
    image (numpy.ndarray): The input binary image (2D array).

    Returns:
    numpy.ndarray: A binary image containing only the largest connected component.
    """
    labeled_image, num_features = label(image, connectivity=2, return_num=True)
    if num_features == 0:
        return image
    regions = regionprops(labeled_image)
    largest_region = max(regions, key=lambda r: r.area)
    largest_component = (labeled_image == largest_region.label)
    return largest_component

# Function to create links shapefile
def create_links(image, metadata):
    """
    Identifies and creates links between adjacent pixels in a binary image. Links are represented as LineStrings.

    Parameters:
    image (numpy.ndarray): The input binary image (2D array) to be processed.
    metadata (dict): The metadata of the raster file, including transform information.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the links as LineStrings.
    """
    links = []
    transform = metadata['transform']
    link_id = 1

    # Iterate over each pixel in the image
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row, col] == 1:  # Check if the pixel is part of a segment
                # Identify neighboring pixels that are also part of the segment
                neighbors = [
                    (row + dr, col + dc) 
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] 
                    if 0 <= row + dr < image.shape[0] and 0 <= col + dc < image.shape[1] and image[row + dr, col + dc] == 1
                ]
                # Create LineString for each neighbor
                for nr, nc in neighbors:
                    x1, y1 = xy(transform, row, col)  # Convert pixel coordinates to spatial coordinates
                    x2, y2 = xy(transform, nr, nc)
                    line = LineString([(x1, y1), (x2, y2)])
                    links.append((link_id, line))  # Append link to the list
                    link_id += 1

    # Remove duplicate links by sorting the coordinates of each LineString
    unique_links = []
    seen = set()
    for link in links:
        coords = tuple(sorted(link[1].coords))
        if coords not in seen:
            seen.add(coords)
            unique_links.append(link)

    # Create a GeoDataFrame from the unique links
    gdf = gpd.GeoDataFrame(unique_links, columns=['id', 'geometry'])
    
    # Set the coordinate reference system (CRS)
    gdf.set_crs(epsg=4326, inplace=True)

    return gdf

# Function to filter links
def filter_links(gdf):
    """
    Filters out diagonal links from a GeoDataFrame of line segments, retaining only those
    that are not part of an intersection with horizontal and vertical links.

    Parameters:
    gdf (geopandas.GeoDataFrame): The input GeoDataFrame containing line segments.

    Returns:
    geopandas.GeoDataFrame: A filtered GeoDataFrame with certain diagonal links removed.
    """
    # Function to categorize the line segments
    def categorize_line(row):
        if row['start_point'][1] == row['end_point'][1]:
            return 'horizontal'
        elif row['start_point'][0] == row['end_point'][0]:
            return 'vertical'
        else:
            return 'diagonal'
    
    # Function to extract start and end coordinates of each line segment
    def get_coordinates(geometry):
        start_point = geometry.coords[0]
        end_point = geometry.coords[1]
        return start_point, end_point
    
    # Apply the function to get coordinates and categorize each segment
    gdf[['start_point', 'end_point']] = gdf.apply(lambda row: get_coordinates(row.geometry), axis=1, result_type='expand')
    gdf['category'] = gdf.apply(categorize_line, axis=1)
    
    # Initialize spatial indexes for horizontal and vertical links
    idx_horizontal = rtree_index.Index()
    idx_vertical = rtree_index.Index()
    
    for idx, row in gdf.iterrows():
        if row['category'] == 'horizontal':
            idx_horizontal.insert(idx, row['geometry'].bounds)
        elif row['category'] == 'vertical':
            idx_vertical.insert(idx, row['geometry'].bounds)
    
    diagonals_to_remove = set()
    
    # Loop through each diagonal link
    for index, diag_row in gdf[gdf['category'] == 'diagonal'].iterrows():
        diag_start = diag_row['start_point']
        diag_end = diag_row['end_point']
        diag_bounds = diag_row['geometry'].bounds
        x_coords = {diag_start[0], diag_end[0]}
        y_coords = {diag_start[1], diag_end[1]}
        hor = ver = False
        
        # Find horizontal links intersecting with the diagonal link using spatial index
        for hor_idx in idx_horizontal.intersection(diag_bounds):
            hor_row = gdf.loc[hor_idx]
            hor_start = hor_row['start_point']
            hor_end = hor_row['end_point']
            if (hor_start[1] in y_coords or hor_end[1] in y_coords) and (hor_start[0] in x_coords and hor_end[0] in x_coords):
                hor = True
                break
        
        # Find vertical links intersecting with the diagonal link using spatial index
        for ver_idx in idx_vertical.intersection(diag_bounds):
            ver_row = gdf.loc[ver_idx]
            ver_start = ver_row['start_point']
            ver_end = ver_row['end_point']
            if (ver_start[0] in x_coords or ver_end[0] in x_coords) and (ver_start[1] in y_coords and ver_end[1] in y_coords):
                ver = True
                break
        
        # Mark the diagonal for removal if it satisfies both conditions
        if hor and ver:
            diagonals_to_remove.add(index)
    
    # Drop the identified diagonal links
    filtered_links = gdf.drop(index=diagonals_to_remove)
    
    # Drop the unnecessary columns before returning
    filtered_links = filtered_links.drop(columns=['start_point', 'end_point', 'category'])
    
    return filtered_links

def remove_degree_2_nodes(G):
    if not isinstance(G, nx.MultiGraph):
        G = nx.MultiGraph(G)

    degree_2_nodes = [node for node, degree in dict(G.degree()).items() if degree == 2]

    for node in degree_2_nodes:
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 2:
            u, v = neighbors
            key_uv = list(G[u][node])[0]
            key_vu = list(G[v][node])[0]
            merged_line = linemerge([G.edges[node, u, key_uv]['geometry'], G.edges[node, v, key_vu]['geometry']])
            G.add_edge(u, v, geometry=merged_line)
            G.remove_node(node)
    
    return G

def geodataframe_to_graph(filtered_links):
    G = nx.MultiGraph()
    
    for idx, row in filtered_links.iterrows():
        line = row.geometry
        start, end = line.coords[0], line.coords[-1]
        G.add_edge(start, end, index=idx, geometry=line)
    
    return G

def graph_to_merged_geodataframes(G):
    nodes = [Point(x, y) for x, y in G.nodes]
    merged_lines = []
    
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        lines = [data['geometry'] for u, v, data in subgraph.edges(data=True)]
        merged_line = unary_union(lines)
        if merged_line.geom_type == 'MultiLineString':
            for line in merged_line.geoms:
                merged_lines.append(line)
        else:
            merged_lines.append(merged_line)
    
    nodes_gdf = gpd.GeoDataFrame(geometry=nodes)
    edges_gdf = gpd.GeoDataFrame(geometry=merged_lines)
    
    return nodes_gdf, edges_gdf

# Function to find furthest endpoints
def find_furthest_endpoints(gdf_points):
    """
    Finds the two furthest nodes in the geodataframe, which may be of type 'endpoint' or 'junction'.

    Parameters:
    gdf_points (geopandas.GeoDataFrame): The geodataframe of points (nodes).

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the two furthest points.
    """
    if len(gdf_points) < 2:
        raise ValueError("Not enough points to find the furthest pair.")
    
    max_distance = 0
    furthest_pair = None
    for (idx1, point1), (idx2, point2) in combinations(gdf_points.iterrows(), 2):
        distance = point1.geometry.distance(point2.geometry)
        if distance > max_distance:
            max_distance = distance
            furthest_pair = (point1, point2)
    
    furthest_geometries = [furthest_pair[0].geometry, furthest_pair[1].geometry]
    start_end_pts = gpd.GeoDataFrame(geometry=furthest_geometries, crs=gdf_points.crs)
    return start_end_pts

def remove_spurs(merged_gdf, start_end_pts):
    start_point = start_end_pts.geometry.iloc[0]
    end_point = start_end_pts.geometry.iloc[1]
    
    G = nx.MultiGraph()
    
    for idx, row in merged_gdf.iterrows():
        line = row.geometry
        start, end = line.coords[0], line.coords[-1]
        G.add_edge(start, end, index=idx, geometry=line)
    
    dead_end_segments = []
    for node in G.nodes:
        if G.degree(node) == 1 and Point(node) not in [start_point, end_point]:
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor = neighbors[0]
                edge_data = G.get_edge_data(node, neighbor)
                for key, data in edge_data.items():
                    dead_end_segments.append(data['index'])
    
    pruned_links = merged_gdf.drop(dead_end_segments)
    
    return pruned_links

def prune_network(edges, start_end_pts):
    """
    Prunes spurs from the network repeatedly until the number of edges remains constant.

    Parameters:
    edges (geopandas.GeoDataFrame): The GeoDataFrame of edges (river segments).
    start_end_pts (geopandas.GeoDataFrame): The GeoDataFrame containing the two furthest points.

    Returns:
    geopandas.GeoDataFrame: A pruned GeoDataFrame with all spurs removed.
    """
    previous_edge_count = -1  # Initialize with an impossible count
    current_edge_count = len(edges)

    while previous_edge_count != current_edge_count:
        previous_edge_count = current_edge_count
        
        # Remove spurs
        edges = remove_spurs(edges, start_end_pts)
        
        # Convert to graph
        G = geodataframe_to_graph(edges)
        
        # Remove degree-2 nodes and merge edges
        G = remove_degree_2_nodes(G)
        
        # Convert back to GeoDataFrame
        _, edges = graph_to_merged_geodataframes(G)
        
        # Update edge count after merging
        current_edge_count = len(edges)
    
    return edges

# Function to find shortest path
def find_shortest_path(start_end_pts, filtered_links):
    """
    Finds the shortest path between two points in a network of filtered links.

    Parameters:
    start_end_pts (geopandas.GeoDataFrame): The GeoDataFrame containing the start and end points.
    filtered_links (geopandas.GeoDataFrame): The GeoDataFrame containing the network of links (line segments).

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the shortest path as a LineString.
    """
    G = nx.Graph()
    for idx, row in filtered_links.iterrows():
        line = row.geometry
        for i in range(len(line.coords) - 1):
            start = Point(line.coords[i])
            end = Point(line.coords[i + 1])
            distance = start.distance(end)
            G.add_edge(tuple(start.coords[0]), tuple(end.coords[0]), weight=distance)
    
    start_point = tuple(start_end_pts.geometry.iloc[0].coords[0])
    end_point = tuple(start_end_pts.geometry.iloc[1].coords[0])
    shortest_path = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')
    shortest_path_coords = [Point(coord) for coord in shortest_path]
    shortest_path_line = LineString(shortest_path_coords)
    shortest_path_length = shortest_path_line.length
    shortest_path_gdf = gpd.GeoDataFrame({'geometry': [shortest_path_line]}, crs=filtered_links.crs)
    return shortest_path_gdf

# Function to classify channels
def classify_channels(edges, shortest_path):
    main_channel_line = shortest_path.geometry.iloc[0]

    # Creating 'chnl_cat' column to classify channels
    edges['chnl_cat'] = edges.apply(
        lambda row: 'main_channel' if row.geometry.within(main_channel_line) else 'other', axis=1
    )
    
    # Assigning unique 'chnl_id' to each segment
    edges['chnl_id'] = None
    edges.loc[edges['chnl_cat'] == 'main_channel', 'chnl_id'] = 1
    
    # Assign unique ids for 'other' channels
    other_idx = edges[edges['chnl_cat'] == 'other'].index
    edges.loc[other_idx, 'chnl_id'] = range(2, 2 + len(other_idx))
    
    return edges

# Function to simplify shortest path
def simplify_shortest_path(shortest_path, num_vertices=10):
    """
    Simplifies the shortest path to a specified number of vertices.

    Parameters:
    shortest_path (geopandas.GeoDataFrame): The GeoDataFrame containing the shortest path as a LineString.
    num_vertices (int, optional): The number of vertices for the simplified path. Default is 10.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the simplified shortest path as a LineString.
    """
    original_line = shortest_path.geometry.iloc[0]
    simplified_coords = [
        original_line.interpolate(i / (num_vertices - 1), normalized=True).coords[0] 
        for i in range(num_vertices)
    ]
    simplified_line = LineString(simplified_coords)
    simplified_path_gdf = gpd.GeoDataFrame({'geometry': [simplified_line]}, crs=shortest_path.crs)
    return simplified_path_gdf

# Function to create perpendicular lines
def create_perpendicular_lines(simplified_path, num_lines=10, fraction_length=1/5):
    """
    Creates perpendicular lines along the simplified path at equal intervals.

    Parameters:
    simplified_path (geopandas.GeoDataFrame): A GeoDataFrame containing the simplified path as a LineString.
    num_lines (int): Number of perpendicular lines to create.
    fraction_length (float): Fraction of the total path length for the length of each perpendicular line.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the perpendicular lines.
    """
    # Extract the LineString from the GeoDataFrame
    line = simplified_path.geometry.iloc[0]
    line_length = line.length
    
    # Calculate spacing between perpendicular lines and half the length of each perpendicular line
    spacing = line_length / num_lines
    half_length = (line_length * fraction_length) / 2
    
    # Generate points at equal intervals along the line
    points = [line.interpolate(i * spacing, normalized=False) for i in range(num_lines)]
    
    perpendicular_lines = []
    
    coords = list(line.coords)
    
    for idx, point in enumerate(points):
        # Find the segment that the point falls on
        segment = None
        for i in range(len(coords) - 1):
            segment_line = LineString([coords[i], coords[i+1]])
            if segment_line.project(point) < segment_line.length:
                segment = segment_line
                break
        
        if segment is None:
            print(f"No segment found for point {idx}: {point}")
            continue
        
        # Calculate the perpendicular direction to the segment
        dx = segment.coords[1][0] - segment.coords[0][0]
        dy = segment.coords[1][1] - segment.coords[0][1]
        length = np.sqrt(dx**2 + dy**2)
        perpendicular_direction = (-dy / length, dx / length)
        
        # Calculate the start and end points of the perpendicular line
        start_point = Point(point.x + half_length * perpendicular_direction[0],
                            point.y + half_length * perpendicular_direction[1])
        end_point = Point(point.x - half_length * perpendicular_direction[0],
                          point.y - half_length * perpendicular_direction[1])
        
        # Create the perpendicular line and add it to the list
        perpendicular_line = LineString([start_point, end_point])
        perpendicular_lines.append(perpendicular_line)
    
    # Create a GeoDataFrame from the perpendicular lines
    channel_belt_cross_sections = gpd.GeoDataFrame({'geometry': perpendicular_lines}, crs=simplified_path.crs)
    
    return channel_belt_cross_sections

# Function to calculate channel count index
def calc_channel_count_index(filtered_links, cross_sections):
    """
    Calculates the Channel Count Index (CCI) for a network of links intersecting with cross sections.

    Parameters:
    filtered_links (geopandas.GeoDataFrame): The GeoDataFrame containing the network of links (line segments) with a 'chnl_id' classification.
    cross_sections (geopandas.GeoDataFrame): The GeoDataFrame containing the cross sections.

    Returns:
    tuple: A tuple containing:
        - cci (float): The Channel Count Index.
        - cross_sections (geopandas.GeoDataFrame): The cross sections GeoDataFrame with an additional 'channel_count' column.
    """
    channel_counts = []
    
    for idx, cross_section in cross_sections.iterrows():
        cross_section_geom = cross_section.geometry
        
        # Find the chnl_ids of segments that intersect the cross section
        intersecting_segments = filtered_links[filtered_links.intersects(cross_section_geom)]
        unique_chnl_ids = intersecting_segments['chnl_id'].unique()
        
        # Count the number of unique chnl_ids intersected by this cross section
        channel_count = len(unique_chnl_ids)
        channel_counts.append(channel_count)
    
    cross_sections['channel_count'] = channel_counts
    
    # Calculate the Channel Count Index (CCI)
    cci = sum(channel_counts) / len(channel_counts)
    
    print(f"Channel Count Index (CCI): {cci}")
    
    return cci, cross_sections

# Function to calculate sinuosity
def calc_sinuosity(shortest_path, simplified_path):
    """
    Calculates the sinuosity of a path by comparing the lengths of the shortest path and the simplified path.

    Parameters:
    shortest_path (geopandas.GeoDataFrame): The GeoDataFrame containing the shortest path as a LineString.
    simplified_path (geopandas.GeoDataFrame): The GeoDataFrame containing the simplified path as a LineString.

    Returns:
    float: The sinuosity value, which is the ratio of the shortest path length to the simplified path length.
    """
    shortest_path_line = shortest_path.geometry.iloc[0]
    simplified_path_line = simplified_path.geometry.iloc[0]
    shortest_path_length = shortest_path_line.length
    simplified_path_length = simplified_path_line.length
    sinuosity = shortest_path_length / simplified_path_length
    print(f"Sinuosity: {sinuosity}")
    return sinuosity

# Function to calculate channel form index
def calculate_channel_form_index(sinuosity, cci):
    """
    Calculates the Channel Form Index (CFI) based on sinuosity and Channel Count Index (CCI).

    Parameters:
    sinuosity (float): The sinuosity of the channel.
    cci (float): The Channel Count Index.

    Returns:
    float: The Channel Form Index (CFI).
    """
    cfi = sinuosity / cci
    print(f"Channel Form Index (CFI): {cfi}")
    return cfi

# Main function to process network
def process_network_folder(river, 
                           radius,
                           min_size = 10,
                           year_range="All", 
                           reach_range="All", 
                           num_lines=10, 
                           num_vertices=10, 
                           fraction_length=1/5, 
                           root_input="C:/Users/huckr/Desktop/UCSB/Dissertation/Data/RiverMapping/RiverMasks", 
                           root_output="C:/Users/huckr/Desktop/UCSB/Dissertation/Data/RiverMapping/Channels"):
    """
    Processes a folder containing water mask rasters to extract river channel networks and calculate metrics.
    Also generates a PDF with plots of classified channels, cross-sections, and other elements.

    Parameters:
    river (str): Name of the river.
    radius (int): Radius for conditional dilation.
    year_range (tuple or str): Year range for processing (default is "All").
    reach_range (tuple or str): Reach range for processing (default is "All").
    num_lines (int): Number of perpendicular lines (cross-sections) (default is 10).
    num_vertices (int): Number of vertices for simplifying the shortest path (default is 10).
    fraction_length (float): Fraction length for creating cross-sections (default is 1/5).
    root_input (str): Root input directory (default is the specified path).
    root_output (str): Root output directory (default is the specified path).

    Returns:
    None
    """
    input_folder = os.path.join(root_input, river)
    output_folder_base = os.path.join(root_output, river)
    
    if year_range == "All":
        year_start, year_end = 1000, 9999  # Arbitrary wide range to include all years
    elif isinstance(year_range, int):
        year_start, year_end = year_range, year_range
    elif isinstance(year_range, tuple):
        year_start, year_end = year_range
    else:
        raise ValueError("Invalid year_range input")
    
    if reach_range == "All":
        reach_start, reach_end = 1, 9999  # Arbitrary wide range to include all reaches
    elif isinstance(reach_range, int):
        reach_start, reach_end = reach_range, reach_range
    elif isinstance(reach_range, tuple):
        reach_start, reach_end = reach_range
    else:
        raise ValueError("Invalid reach_range input")
    
    # Initialize a dictionary to store metrics
    metrics = {}

    # Create a PDF file to store the plots
    pdf_path = os.path.join(output_folder_base, f'{river}_report.pdf')
    with PdfPages(pdf_path) as pdf:
        # Title page with summary information
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        summary_text = (f"River Name: {river}\n"
                        f"Year Range: {year_start} - {year_end}\n"
                        f"Reach Range: {reach_start} - {reach_end}\n"
                        f"Radius for Conditional Dilation: {radius}\n"
                        f"Minimum Size for Islands: {min_size}\n"
                        f"Number of Perpendicular Lines: {num_lines}\n"
                        f"Number of Vertices for Simplification: {num_vertices}\n"
                        f"Fraction Length for Cross-Sections: {fraction_length}\n")
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        # Process each reach folder
        for reach_folder in glob(os.path.join(input_folder, 'reach_*')):
            reach_folder_name = os.path.basename(reach_folder)
            match_reach = re.match(r"reach_(\d+)", reach_folder_name)
            if match_reach:
                reach = int(match_reach.group(1))
                if reach_start <= reach <= reach_end:
                    processed_folder = os.path.join(reach_folder, 'Processed')
                    for file_path in glob(os.path.join(processed_folder, '*.tif')):
                        file_name = os.path.basename(file_path)
                        match_year = re.match(rf"{river}_reach_{reach}_(\d{{4}})_.*\.tif", file_name)
                        if match_year:
                            year = int(match_year.group(1))
                            if year_start <= year <= year_end:
                                output_folder = os.path.join(output_folder_base, f"reach_{reach}", str(year))
                                os.makedirs(output_folder, exist_ok=True)
                                
                                try:
                                    water_mask, metadata = load_raster(file_path)
                                    cleaned_water_mask = eliminate_small_islands(water_mask, min_size=10)
                                    skeleton = skeletonize(cleaned_water_mask > 0)
                                    dilated_skeleton = conditional_dilation(skeleton, radius)
                                    reskeletonized = skeletonize(dilated_skeleton > 0)
                                    largest_component = keep_largest_component(reskeletonized)
                                    
                                    largest_component_output_path = os.path.join(output_folder, 'largest_component.tif')
                                    save_raster(largest_component_output_path, largest_component, metadata)

                                    initial_links = create_links(largest_component, metadata)
                                    filtered_links = filter_links(initial_links)
                                    
                                    chan_graph1 = geodataframe_to_graph(filtered_links)

                                    chan_graph2 = remove_degree_2_nodes(chan_graph1)

                                    nodes, edges = graph_to_merged_geodataframes(chan_graph2)
                                    start_end_pts = find_furthest_endpoints(nodes)
                                    pruned_edges = prune_network(edges, start_end_pts)

                                    shortest_path_gdf = find_shortest_path(start_end_pts, pruned_edges)
                                    classified_links = classify_channels(pruned_edges, shortest_path_gdf)
                                    valley_center_line = simplify_shortest_path(shortest_path_gdf, num_vertices)
                                    channel_belt_cross_sections = create_perpendicular_lines(valley_center_line, num_lines, fraction_length)
                                    
                                    sinuosity_value = calc_sinuosity(shortest_path_gdf, valley_center_line)
                                    cci, updated_cross_sections = calc_channel_count_index(classified_links, channel_belt_cross_sections)
                                    cfi_value = calculate_channel_form_index(sinuosity_value, cci)
                                    
                                    classified_links.to_file(os.path.join(output_folder, 'channel_links.shp'))
                                    channel_belt_cross_sections.to_file(os.path.join(output_folder, 'channel_belt_cross_sections.shp'))
                                    nodes.to_file(os.path.join(output_folder, 'nodes.shp'))
                                    shortest_path_gdf.to_file(os.path.join(output_folder, 'main_channel.shp'))
                                    valley_center_line.to_file(os.path.join(output_folder, 'valley_center_line.shp'))
                                    
                                    # Store metrics
                                    reach_key = f"reach_{reach}"
                                    if reach_key not in metrics:
                                        metrics[reach_key] = {}
                                    metrics[reach_key][year] = {
                                        'Sinuosity': sinuosity_value,
                                        'CCI': cci,
                                        'CFI': cfi_value
                                    }

                                    # Generate a plot for the PDF
                                    fig, ax = plt.subplots(figsize=(8.5, 11))
                                    ax.set_title(f"Reach {reach}, Year {year}")
                                    
                                    # Plot the cleaned water mask at the bottom
                                    show(cleaned_water_mask, transform=metadata['transform'], ax=ax, cmap='gray')
                                    
                                    # Plot classified channels and cross-sections
                                    classified_links.plot(ax=ax, color='#39FF14', linewidth=1)
                                    channel_belt_cross_sections.plot(ax=ax, color='orange', linewidth=1)
                                    
                                    # Plot the main channel on top
                                    shortest_path_gdf.plot(ax=ax, color='red', linewidth=2)
                                    
                                    # Add channel counts at the end of each cross-section
                                    for idx, row in updated_cross_sections.iterrows():
                                        x, y = row.geometry.centroid.x, row.geometry.centroid.y
                                        ax.text(x, y, str(row['channel_count']), fontsize=8, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))

                                    # Display metrics in the bottom right corner
                                    ax.text(0.95, 0.05, f"Sinuosity: {sinuosity_value:.2f}\nCCI: {cci:.2f}\nCFI: {cfi_value:.2f}",
                                            ha='right', va='bottom', transform=ax.transAxes, fontsize=10,
                                            bbox=dict(facecolor='white', alpha=0.5))
                                    
                                    # Save the plot to the PDF
                                    pdf.savefig(fig)
                                    plt.close(fig)

                                except Exception as e:
                                    logging.error(f"Error processing file {file_path}: {e}")
                                    continue

        # Save metrics to an Excel workbook
        metrics_output_path = os.path.join(output_folder_base, f'{river}_metrics.xlsx')
        with pd.ExcelWriter(metrics_output_path) as writer:
            for reach, reach_metrics in metrics.items():
                df = pd.DataFrame.from_dict(reach_metrics, orient='index')
                df.to_excel(writer, sheet_name=reach)
                
def main(input_directory):
    """
    Main function to process rivers based on a CSV file of input variables.
    
    Args:
        input_directory (str): The directory where the input .csv file resides.
    
    The .csv file should contain the following columns:
        - river_name
        - radius
        - min_blob_size
        - year_range
        - reach_range
        - num_xcs (num_lines)
        - num_vertices
        - fraction_length
        - root_input
        - root_output
    """
    
    # Load the CSV into a pandas DataFrame
    river_data = pd.read_csv(input_directory)
    
    # Iterate through each row (each river) and run the process_network_folder() function
    for index, row in river_data.iterrows():
        river_name = row['river_name']
        radius = row['dilation_radius']
        min_blob_size = row['min_blob_size']
        year_range = eval(row['year_range'])  # Convert string representation to tuple or value
        reach_range = eval(row['reach_range'])  # Convert string representation to tuple or value
        num_lines = row['num_xcs']
        num_vertices = row['num_vertices']
        fraction_length = float(Fraction(row['fraction_length']))
        root_input = row['mask_output_path']
        root_output = row['channel_files_output']

        print(f"Processing river: {river_name}")
        
        # Call the existing function with inputs from the current row
        process_network_folder(
            river=river_name,
            radius=radius,
            min_size=min_blob_size,
            year_range=year_range,
            reach_range=reach_range,
            num_lines=num_lines,
            num_vertices=num_vertices,
            fraction_length=fraction_length,
            root_input=root_input,
            root_output=root_output
        )
        
    print("All rivers processed.")

csv_path = "/home/jamesrees/Desktop/RivMapper/Data/RiverMapping/Bermejo_river_datasheet.csv"

main(csv_path)