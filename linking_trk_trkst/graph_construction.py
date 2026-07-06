import sys
import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Dataset, Data

def construct_graphs(event, data_trk, data_trkst, event_nodes, node_feature_trkst, node_feature_trk):
    try:
        # Trackster truth labels
        ts_labels_full = ak.to_numpy(data_trkst["y"][event])
        #print(" ts_labels_full ",len(ts_labels_full))
        # Find valid tracks (i.e. not tracksters and have neighbours)
        valid_tracks = [(event, node.index, node)
                        for trk_idx, node in enumerate(event_nodes)
                        if (not node.is_trackster and len(node.neighbours) > 0)]

        #if len(valid_tracks) == 0:
        #    continue
        N_trk = len(valid_tracks)        #print(" valid_tracks ", valid_tracks)
        # Collect all used trackster indices
        '''        
        used_ts_indices = set()
        for _, _, node in valid_tracks:
            print("Node index:", node.index)
            print("is_trackster:", node.is_trackster)
            print("already_visited:", node.already_visited)
            print("neighbours:", node.neighbours)
            used_ts_indices.update(node.neighbours)
            print(" used_ts_indices ", used_ts_indices)
        '''
        used_ts_indices = []
        seen = set()
        for _, _, node in valid_tracks:
            for ts_idx in node.neighbours:
                if ts_idx not in seen:
                    used_ts_indices.append(ts_idx)
                    seen.add(ts_idx)
        #rint(" used_ts_indices ", used_ts_indices)         
        # Map from original trackster index -> new index in graph
        #ts_index_map = {ts_idx: i for i, ts_idx in enumerate(sorted(used_ts_indices))}
        ts_index_map = {ts_idx: i for i, ts_idx in enumerate(used_ts_indices)}
        ts_index_inv_map = {v: k for k, v in ts_index_map.items()}  # inverse map
        N_ts = len(ts_index_map)
        N_nodes = N_trk + N_ts

        # Build edges
        edges = [[], []]
        for new_trk_idx, (ev_idx, trk_idx, node) in enumerate(valid_tracks):
            for ts_local_idx in node.neighbours:
                if ts_local_idx in ts_index_map:
                    ts_new_idx = N_trk + ts_index_map[ts_local_idx]
                    edges[0].append(new_trk_idx)
                    edges[1].append(ts_new_idx)


        # Build node features
        feature_dim = len(node_feature_trk) + len(node_feature_trkst)
        nodes = np.zeros((N_nodes, feature_dim))

        # Fill track features
        for i, key in enumerate(node_feature_trk):
            for new_idx, (ev_idx, original_idx, _) in enumerate(valid_tracks):
                #print(" track new_idx ", new_idx, " track ", original_idx)
                # use ev_idx here, not outer event variable 
                full_values = ak.to_numpy(data_trk[ev_idx][key])  
                nodes[new_idx, i] = full_values[original_idx]

        # Fill trackster features
        for j, key in enumerate(node_feature_trkst):
            full_values = ak.to_numpy(data_trkst[event][key])
            for ts_orig_idx, ts_new_idx in ts_index_map.items():
                #print(" ts_orig_idx ", ts_orig_idx, " ts_new_idx ", ts_new_idx)
                nodes[N_trk + ts_new_idx, j + len(node_feature_trk)] = full_values[ts_orig_idx]
        # Edge features
        #edges_np = np.array(edges)
        #num_edges = edges_np.shape[1] if edges_np.ndim == 2 else len(edges[0])
        #num_edge_features = 6
        #edge_features = np.zeros((num_edges, num_edge_features))
        # strating of nodes[edges[1] features is 10 tracksters while for nodes[edges[0] is 0 tracks
        #print( " nodes[edges[1], 9] ", nodes[edges[1], 9])
        #print( " nodes[edges[0], 0] ", nodes[edges[0], 0])        
        #edge_features[:, 0] = np.abs(nodes[edges[1], 0] - nodes[edges[0], 0])
        #edge_features[:, 1] = np.abs(nodes[edges[1], 1] - nodes[edges[0], 1])
        #edge_features[:, 2] = np.linalg.norm(nodes[edges[1], :5] - nodes[edges[0], :5], axis=1)
        #print(" edges ", edges)
        # edges: tuple (source indices, target indices)
        edges_np = np.array(edges)
        num_edges = edges_np.shape[1] if edges_np.ndim == 2 else len(edges[0])
        num_edge_features = 11
        edge_features = np.zeros((num_edges, num_edge_features))
        # source = tracks (indices: 0–9)
        # target = tracksters (indices: 10–20)
        src = edges[0]  # tracks
        tgt = edges[1]  # tracksters
        # Trackster feature offset
        trkster_offset = 10
        # 1 deltaEta , deltaPhi 
        deta =  nodes[src, 0] - nodes[tgt, trkster_offset + 0]
        dphi =  nodes[src, 1] - nodes[tgt, trkster_offset + 1] 
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        edge_features[:, 0] = deta
        edge_features[:, 1] = dphi

        # 2 deltaEta/ phi significance
        deta_sig = deta / np.sqrt(nodes[tgt, trkster_offset + 2]**2 + nodes[src, 2]**2 + 1e-8)
        dphi_sig = dphi / np.sqrt(nodes[tgt, trkster_offset + 3]**2 + nodes[src, 3]**2 + 1e-8)
        edge_features[:, 2] = deta_sig
        edge_features[:, 3] = dphi_sig

        # 3 deltaEta - deltaPhi distance (DeltaR)
        edge_features[:, 4] = np.sqrt(deta**2 + dphi**2)
 
        # 4 3D distance (dx, dy, dz)
        dx = nodes[tgt, trkster_offset + 4] - nodes[src, 4]
        dy = nodes[tgt, trkster_offset + 5] - nodes[src, 5]
        dz = nodes[tgt, trkster_offset + 6] - nodes[src, 6]
        edge_features[:, 5] = np.sqrt(dx**2 + dy**2 + dz**2)

        # xy distance
        edge_features[:, 6] = np.sqrt(dx**2 + dy**2)

        # 5 Energy difference and ratio
        dE = nodes[tgt, trkster_offset + 7] - nodes[src, 7]  # raw_energy - track_p
        E_ratio = nodes[tgt, trkster_offset + 7] / (nodes[src, 7] + 1e-8)
        edge_features[:, 7] = dE
        edge_features[:, 8] = E_ratio

        # 6 Time difference and significance
        #dtime = nodes[tgt, trkster_offset + 8] - nodes[src, 8]
        #time_err_combined = np.sqrt(nodes[tgt, trkster_offset + 9]**2 + nodes[src, 9]**2 + 1e-8)
        #dtime_sig = dtime / time_err_combined
        #edge_features[:, 9] = dtime
        #edge_features[:, 10] = dtime_sig
       	# Full 3D Euclidean distance in 4:7 coordinates (same as 3D distance above)
        #edge_features[:, 11] = np.linalg.norm(nodes[tgt, trkster_offset + 4: trkster_offset + 7] - nodes[src, 4:7], axis=1)

        # 7 Euclidean distance between track MTD hit and trackster barycenter and Time-of-flight 
        #c_cm_per_ns = 29.9792458
        #delta_x = nodes[tgt, trkster_offset + 4] - nodes[src, 12]
        #delta_y = nodes[tgt, trkster_offset + 5] - nodes[src, 13]
        #delta_z = nodes[tgt, trkster_offset + 6] - nodes[src, 14]
        #spatial_distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        #track_beta = nodes[src, 11]
        #beta_speed = 1.0 if track_beta == 0.0 else track_beta * c_cm_per_ns
        #tof = spatial_distance / beta_speed
        #print(" track_beta ", track_beta)
        #tof = spatial_distance / (track_beta * c_cm_per_ns + 1e-8)
        #print(" tof ", tof)        
        #edge_features[:, 11] = spatial_distance
        #edge_features[:, 13] = tof        
        #print(" trackster features ", nodes[tgt, :])
        #print(" track features ", nodes[src, :])

        #print(" trackster features ", nodes[tgt, trkster_offset])

        #print("Edge features shape:", edge_features.shape)
        #print("Example edge feature row:", edge_features[0])

        # Labels: 1 if trackster is matched, 0 otherwise
        y = np.zeros(num_edges)
        for i, (trk_idx, ts_new_idx) in enumerate(zip(edges[0], edges[1])):
            ts_relative_idx = ts_new_idx - N_trk  # shift to 0-based index in tracksters
            ts_orig_idx = ts_index_inv_map[ts_relative_idx]  # get back original index
            #print(" ts_orig_idx ", ts_orig_idx)
            
            ts_label = ts_labels_full[ts_orig_idx]
            y[i] = 1 if ts_label != -1 else 0

        # Final graph
        graph = Data(
            x=torch.tensor(nodes, dtype=torch.float32),
            edge_index=torch.tensor(edges, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            num_nodes=N_nodes,
        )
        #print(" graph.x ",graph.x)
        #print(" graph.edge_index ",graph.edge_index)
        #print(" graph.edge_attr ",graph.edge_attr)
        #print(" graph.y ",graph.y)
        #print(" graph.num_nodes ",graph.num_nodes)
        #print(" =====================================End of Graph Inspection ====================================")
        '''
        # Optional debug: check masks
        x = graph.x
        split_idx = len(node_feature_trk)
        is_track     = (x[:, split_idx:] == 0).all(dim=1)
        is_trackster = (x[:, :split_idx] == 0).all(dim=1)
        track_nodes     = x[is_track]
        trackster_nodes = x[is_trackster]

        print("track_nodes:\n", track_nodes)
        print("trackster_nodes:\n", trackster_nodes)
        '''
        return graph

    except Exception as e:
        print(f"Skipping event {event} due to error: {e}")
        return None
