# c:\Users\Kevin\Github\vote-estimator\visualize_in_qgis.py
import argparse
import os
import pickle
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from tqdm import tqdm

def export_graph_to_shapefiles(graph_file, nodes_out, edges_out, crs="EPSG:3857"):
    """
    Loads a NetworkX graph from a gpickle file and exports its nodes and edges
    as separate shapefiles for use in GIS software.

    Args:
        graph_file (str): Path to the input .gpickle file.
        nodes_out (str): Path for the output nodes shapefile.
        edges_out (str): Path for the output edges shapefile.
        crs (str): The Coordinate Reference System to use for the output shapefiles.
    """
    print(f"Loading graph from {graph_file}...")
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    if not G.nodes:
        print("Graph is empty. Exiting.")
        return

    # --- Export Nodes ---
    print(f"Exporting {G.number_of_nodes()} nodes to {nodes_out}...")
    nodes_data = []
    # The original polygon geometry is stored as a node attribute. We'll use its centroid.
    for node_id, attributes in tqdm(G.nodes(data=True), desc="Processing nodes"):
        if 'geometry' in attributes:
            centroid = attributes['geometry'].centroid
            nodes_data.append({'node_id': node_id, 'geometry': centroid})
        else:
            print(f"Warning: Node {node_id} is missing 'geometry' attribute. Skipping.")

    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=crs)
    nodes_gdf.to_file(nodes_out, driver='ESRI Shapefile')

    # --- Export Edges ---
    print(f"Exporting {G.number_of_edges()} edges to {edges_out}...")
    # Create a quick lookup for node centroids
    centroid_map = nodes_gdf.set_index('node_id')['geometry']
    
    edges_data = []
    for u, v in tqdm(G.edges(), desc="Processing edges"):
        p1 = centroid_map.get(u)
        p2 = centroid_map.get(v)
        if p1 and p2:
            line = LineString([p1, p2])
            edges_data.append({'source': u, 'target': v, 'geometry': line})

    edges_gdf = gpd.GeoDataFrame(edges_data, crs=crs)
    edges_gdf.to_file(edges_out, driver='ESRI Shapefile')

    print("\nExport complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert a NetworkX gpickle file to shapefiles for QGIS.")
    parser.add_argument("graph_file", help="Path to the input .gpickle file (e.g., blockgroups_graph.gpickle).")
    parser.add_argument("--nodes_out", default="output/graph_nodes.shp", help="Path for the output nodes shapefile.")
    parser.add_argument("--edges_out", default="output/graph_edges.shp", help="Path for the output edges shapefile.")
    args = parser.parse_args()

    for p in (args.nodes_out, args.edges_out):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
    export_graph_to_shapefiles(args.graph_file, args.nodes_out, args.edges_out)

if __name__ == "__main__":
    main()
