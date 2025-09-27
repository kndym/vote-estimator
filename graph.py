"""
Build a rook adjacency graph from block group shapefile.

- Input: block_groups.shp (polygon shapefile of block groups)
- Output:
    - blockgroups_graph.gpickle (NetworkX graph object)
    - adjacency_edges.csv (edge list)
    - quick_plot.png (visual check)

Each node = block group, each edge = rook adjacency (shared boundary).
Optimized for >10,000 polygons using spatial join.
"""

import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from tqdm import tqdm
import argparse
import pickle


def build_rook_graph(gdf, simplify_tolerance=None):
    """
    Build rook adjacency graph from a GeoDataFrame.
    Args:
        gdf (GeoDataFrame): GeoDataFrame of polygons, in EPSG:3857.
        simplify_tolerance (float or None): optional simplify tolerance in CRS units
    Returns:
        G (networkx.Graph), gdf (GeoDataFrame with GEOID), neighbors (GeoDataFrame of edges)
    """
    print("Building rook graph...")
    print("[1/7] Initializing and identifying GEOID...")
    # Try to find a GEOID column, otherwise fall back to index
    if 'GEOID20' in gdf.columns:
        print("Using 'GEOID20' column as node identifier.")
        gdf["GEOID"] = gdf["GEOID20"].astype(str)
    elif 'GEOID10' in gdf.columns:
        print("Using 'GEOID10' column as node identifier.")
        gdf["GEOID"] = gdf["GEOID10"].astype(str)
    elif 'GEOID' in gdf.columns:
        print("Using 'GEOID' column as node identifier.")
        gdf["GEOID"] = gdf["GEOID"].astype(str)
    else:
        print("Warning: No standard GEOID column found (e.g., GEOID20, GEOID10, GEOID).")
        print("Falling back to using the GeoDataFrame's index as the node identifier.")
        print("If the index is not the actual geographic identifier, the graph will not match the simulation data.")
        gdf["GEOID"] = gdf.index.astype(str)

    # Optional simplification (to speed up large files)
    print("[2/7] Simplifying geometries...")
    if simplify_tolerance:
        gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance)

    # Check for and fix invalid geometries which can cause TopologyException
    print("[3/7] Fixing invalid geometries...")
    invalid_mask = ~gdf.is_valid
    if invalid_mask.any():
        print(f"Found {invalid_mask.sum()} invalid geometries. Attempting to fix with .buffer(0)...")
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)

        # Re-check after fixing
        still_invalid_mask = ~gdf.is_valid
        if still_invalid_mask.any():
            print(f"Warning: {still_invalid_mask.sum()} geometries are still invalid after attempting to fix.")
        else:
            print("All geometries are now valid.")

    # Use vectorized spatial join for touching polygons
    print("[4/7] Performing spatial join (this may take a while)...")
    neighbors = gpd.sjoin(
        gdf[["GEOID", "geometry"]],
        gdf[["GEOID", "geometry"]],
        how="inner",
        predicate="touches"
    )

    # Remove self-joins and duplicates
    print("[5/7] Filtering edges...")
    neighbors = neighbors[neighbors["GEOID_left"] < neighbors["GEOID_right"]]

    # Build graph
    print("[6/7] Building graph...")
    G = nx.from_pandas_edgelist(neighbors, "GEOID_left", "GEOID_right")

    # Attach geometry to nodes
    print("[7/7] Attaching node attributes...")
    geom_dict = gdf.set_index("GEOID")["geometry"].to_dict()
    nx.set_node_attributes(G, geom_dict, "geometry")

    return G, gdf, neighbors


def quick_plot(gdf, edges, outpath="quick_plot.png"):
    """Simple visualization of adjacency graph"""
    print(f"Generating plot and saving to {outpath}...")
    ax = gdf.plot(edgecolor="lightgray", facecolor="white", figsize=(10, 10))

    # Create a list of line segments from the centroids of adjacent polygons
    centroids = gdf.geometry.centroid
    edge_iterator = tqdm(
        zip(edges["GEOID_left"], edges["GEOID_right"]),
        total=len(edges),
        desc="Preparing plot edges"
    )
    edge_coords = [
        [(centroids.iloc[u].x, centroids.iloc[u].y), (centroids.iloc[v].x, centroids.iloc[v].y)]
        for u, v in edge_iterator
    ]

    # Plot edges and centroids
    lines = mc.LineCollection(edge_coords, color="red", linewidth=0.3, alpha=0.6)
    ax.add_collection(lines)
    centroids.plot(ax=ax, color="blue", markersize=5)

    plt.title("Rook Adjacency Graph (centroids + edges)")
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build a rook adjacency graph from a polygon shapefile.")
    parser.add_argument("shapefile", help="Path to the input polygon shapefile.")
    parser.add_argument("--simplify", type=float, default=None, help="Optional simplification tolerance in CRS units.")
    parser.add_argument("--graph_out", default="blockgroups_graph.gpickle", help="Path for the output graph file.")
    parser.add_argument("--edges_out", default="adjacency_edges.csv", help="Path for the output edges CSV file.")
    parser.add_argument("--plot_out", default="quick_plot.png", help="Path for the output plot image.")
    parser.add_argument("--map_out", default="input_map.png", help="Path for the output initial map image.")
    args = parser.parse_args()

    # Load shapefile and set CRS
    print("Loading shapefile...")
    gdf = gpd.read_file(args.shapefile).to_crs(epsg=3857)
    print(f"Loaded {len(gdf)} polygons.")

    G, gdf, neighbors = build_rook_graph(gdf, simplify_tolerance=args.simplify)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Save outputs
    print(f"Saving graph to {args.graph_out}...")
    with open(args.graph_out, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saving edges to {args.edges_out}...")
    neighbors.to_csv(args.edges_out, index=False)

    print(f"\nOutputs saved:\n- Initial Map: {args.map_out}\n- Graph: {args.graph_out}\n- Edges: {args.edges_out}\n- Plot: {args.plot_out}")


if __name__ == "__main__":
    main()