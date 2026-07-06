from scipy.spatial import KDTree
import numpy as np
import torch
import awkward as ak

def adding_addFeatures(event, graph, data_trkst, all_nodes, node_feature_keys_before_trk, node_feature_keys_before_trkst):
    try:
        ts_labels_full = ak.to_numpy(data_trkst["y"][event])

        # Strictly following your updated valid_tracks logic:
        valid_tracks = [
            (event, node.index, node)
            for trk_idx, node in enumerate(all_nodes)
            if (not node.is_trackster and len(node.neighbours) > 0)
        ]
        N_trk = len(valid_tracks)
        if N_trk == 0:
            return None

        # Step 1: Get only the trackster indices actually connected
        #used_ts_indices = set()
        #for _, _, node in valid_tracks:
        #    used_ts_indices.update(node.neighbours)

        used_ts_indices = []
        seen = set()
        for _, _, node in valid_tracks:
            for ts_idx in node.neighbours:
                if ts_idx not in seen:
                    used_ts_indices.append(ts_idx)
                    seen.add(ts_idx)
        ts_index_map = {ts_idx: i for i, ts_idx in enumerate(used_ts_indices)}
        #ts_index_map = {ts_idx: i for i, ts_idx in enumerate(sorted(used_ts_indices))}
        N_ts = len(ts_index_map)

        # Step 2: Build edge index map
        edges = graph.edge_index.numpy()
        num_edges = edges.shape[1]
        edge_indices = -1 * np.ones((N_trk, N_ts), dtype=np.int64)
        for i in range(num_edges):
            src, dst = edges[0, i], edges[1, i]
            if src < N_trk and dst >= N_trk:
                ts_local_idx = dst - N_trk
                edge_indices[src, ts_local_idx] = i
        # Step 3: Extend edge features to 6 columns
        edge_features = graph.edge_attr.numpy()
        if edge_features.shape[1] < 11:
            new_edge_features = np.zeros((num_edges, 11), dtype=edge_features.dtype)
            new_edge_features[:, :edge_features.shape[1]] = edge_features
            edge_features = new_edge_features
        '''
        # Find index of barycenter_z in trackster features
        barycenter_z_index = len(node_feature_keys_before_trk) + node_feature_keys_before_trkst.index("barycenter_z")
        print(f"\nEvent {event} - barycenter_z of connected tracksters:")
        for i in range(num_edges):
            src, dst = edges[0, i], edges[1, i]
            if src < N_trk and dst >= N_trk:
                bary_z = graph.x[dst, barycenter_z_index].item()
                print(f"  Track {src} -> Trackster {dst - N_trk}, barycenter_z = {bary_z:.2f}")
        '''
        # Step 4: Compute new edge features
        for new_trk_idx, (ev_idx, original_idx, node) in enumerate(valid_tracks):
            track_pos = graph.x[new_trk_idx, 4:7].numpy()  # track (x, y, z)
            #print(track_pos)
            for ts_orig_idx, ts_local_idx in ts_index_map.items():
                edge_idx = edge_indices[new_trk_idx, ts_local_idx]
                if edge_idx == -1:
                    continue

                ts_vertices = np.asarray(data_trkst["vertices"][event][ts_orig_idx])
                ts_vertices[:, 2] = np.abs(ts_vertices[:, 2])  # Make z component positive
                if len(ts_vertices) == 0:
                    edge_features[edge_idx, 9:11] = 0
                    continue

                dists = np.linalg.norm(ts_vertices - track_pos, axis=1)
                edge_features[edge_idx, 9] = np.min(dists)
                edge_features[edge_idx, 10] = np.max(dists)

                # Count nearby vertices within radius=1
                #radius = 1.0
                #tree = KDTree(ts_vertices)
                #nearby_count = len(tree.query_ball_point(track_pos, radius))
                #edge_features[edge_idx, 16] = nearby_count

        # Update edge features in the graph
        graph.edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        print("Edge features shape:", graph.edge_attr.shape)
        #print("Example edge feature row:", graph.edge_attr[0])

        print(" graph.x ",graph.x)
        print(" graph.edge_index ",graph.edge_index)
        print(" graph.edge_attr ",graph.edge_attr)
        print(" graph.y ",graph.y)
        print(" graph.num_nodes ",graph.num_nodes)
        #print(" =====================================End of Graph Inspection ====================================")
        return graph

    except Exception as e:
        print(f"Skipping event {event} due to error: {e}")
        return None

