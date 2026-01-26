import networkx as nx
import pickle
import argparse

def inspect_gpickle(graph_file, num_samples=5):
    """
    Loads a NetworkX graph from a gpickle file and prints a sample of its contents.

    Args:
        graph_file (str): Path to the input .gpickle file.
        num_samples (int): The number of sample nodes and edges to print.
    """
    print(f"Loading and inspecting graph from {graph_file}...")
    try:
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {graph_file}")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    if not isinstance(G, nx.Graph):
        print(f"The loaded object is not a NetworkX graph, but a {type(G)}.")
        return

    print("\n--- Graph Summary ---")
    print(f"Graph Type: {type(G)}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print("\nGraph is empty.")
        return

    # --- Print Sample Nodes ---
    print(f"\n--- Sample Nodes (first {num_samples}) ---")
    node_ids = list(G.nodes())
    for node_id in node_ids[:num_samples]:
        attributes = G.nodes[node_id]
        print(f"Node '{node_id}':")
        for key, value in attributes.items():
            # The 'geometry' attribute can be very long, so we'll just print its type.
            if key == 'geometry':
                print(f"  - {key}: <{type(value).__name__}>")
            else:
                print(f"  - {key}: {value}")

    # --- Print Sample Edges ---
    if G.number_of_edges() > 0:
        print(f"\n--- Sample Edges (first {num_samples}) ---")
        for u, v in list(G.edges())[:num_samples]:
            print(f"Edge: ({u}, {v})")

    print("\nInspection complete.")

def main():
    parser = argparse.ArgumentParser(description="Inspect a NetworkX gpickle file.")
    parser.add_argument("graph_file", help="Path to the input .gpickle file (e.g., blockgroups_graph.gpickle).")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample nodes/edges to show.")
    args = parser.parse_args()

    inspect_gpickle(args.graph_file, num_samples=args.samples)

if __name__ == "__main__":
    main()