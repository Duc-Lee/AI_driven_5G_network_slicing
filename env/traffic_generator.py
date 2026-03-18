import numpy as np
import json
import os
import networkx as nx
import math

class TrafficGenerator:
    def __init__(self, slice_names, data_dir="dataset"):
        self.slice_names = slice_names
        self.data_dir = data_dir
        self.slices_dir = os.path.join(data_dir, "slices")
        self.graphs_dir = os.path.join(data_dir, "graphs")
        self.time_step = 0
        if os.path.exists(self.slices_dir):
            self.files = sorted([f for f in os.listdir(self.slices_dir) if f.startswith("slices_") and f.endswith(".json")], key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            self.files = []
        self.num_files = len(self.files)
        self.cached_graphs = {}
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        # earth radius (m)
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _get_graph(self, file_idx):
        if file_idx not in self.cached_graphs:
            graph_file = os.path.join(self.graphs_dir, f"graph_{file_idx}.txt")
            if os.path.exists(graph_file):
                try:
                    self.cached_graphs[file_idx] = nx.read_gml(graph_file, label='id')
                except Exception as e:
                    print(f"Error loading graph {graph_file}: {e}")
                    self.cached_graphs[file_idx] = None
            else:
                self.cached_graphs[file_idx] = None
        return self.cached_graphs[file_idx]

    def generate_traffic(self):
    # generate UE traffic based on dataset
        traffic_data = {name: [] for name in self.slice_names}
        if self.num_files == 0:
            self.time_step += 1
            return traffic_data
        file_idx = self.time_step % self.num_files
        self.time_step += 1
        file_path = os.path.join(self.slices_dir, self.files[file_idx])
        graph = self._get_graph(file_idx)
        with open(file_path, 'r') as f:
            data = json.load(f)
        for slice_data in data:
            slice_type = slice_data.get('type')
            if slice_type in self.slice_names:
                for flow in slice_data.get('flows', []):
                    demand_mbps = flow.get('bandwidth', 0) / 1000000.0
                    origin = flow.get('origin_node')
                    antenna = flow.get('origin_node_antenna')
                    distance = 100.0
                    if graph is not None and origin is not None and antenna is not None:
                        if origin in graph.nodes and antenna in graph.nodes:
                            node_o = graph.nodes[origin]
                            node_a = graph.nodes[antenna]
                            if 'Latitude' in node_o and 'Longitude' in node_o and \
                            'Latitude' in node_a and 'Longitude' in node_a:
                                dist = self._haversine_distance(
                                    node_o['Latitude'], node_o['Longitude'],
                                    node_a['Latitude'], node_a['Longitude']
                                )
                                distance = max(1.0, dist)
                    distance = max(10.0, distance)
                    traffic_data[slice_type].append({
                        'distance': distance,
                        'demand': demand_mbps
                    })
        return traffic_data
