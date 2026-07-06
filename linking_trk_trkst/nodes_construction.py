# graph_construction.py
from collections import defaultdict
import numpy as np
#from EtaPhiTile import EtaPhiTile  # Ensure you have this in your path or module
#from node import Node              # Same here


class EtaPhiTile:
    def __init__(self, eta_bins, phi_bins):
        self.eta_bins = eta_bins
        self.phi_bins = phi_bins
        self.tile = defaultdict(list)

    def _get_bin(self, eta, phi):
        eta_bin = np.digitize([eta], self.eta_bins)[0]
        phi_bin = np.digitize([phi], self.phi_bins)[0]
        return (eta_bin, phi_bin)

    def fill(self, eta, phi, index):
        b = self._get_bin(eta, phi)
        self.tile[b].append(index)

    def query(self, eta, phi, d_eta=1, d_phi=1):
        eta_bin, phi_bin = self._get_bin(eta, phi)
        neighbors = []
        for de in range(-d_eta, d_eta + 1):
            for dp in range(-d_phi, d_phi + 1):
                b = (eta_bin + de, phi_bin + dp)
                neighbors.extend(self.tile.get(b, []))
        return neighbors

class Node:
    def __init__(self, index: int, is_trackster: bool):
        self.index = index
        self.is_trackster = is_trackster
        self.neighbours = []  
        self.already_visited = False

    def add_neighbour(self, other_index):
        self.neighbours.append(other_index)


def construct_nodes(trks, all_tracksters, all_valid_track_indices, tsCP):
    all_events_nodes = []

    for ev in range(len(tsCP)):
        event_nodes = []
        delta = 0.1
        min_eta_pos, max_eta_pos = 1.5, 3.2
        min_eta_neg, max_eta_neg = -3.2, -1.5

        # Set up eta-phi tiling
        eta_edges_pos = np.linspace(min_eta_pos, max_eta_pos, 24)
        eta_edges_neg = np.linspace(min_eta_neg, max_eta_neg, 24)
        phi_edges = np.linspace(-np.pi, np.pi, 126)

        trackTilePos = EtaPhiTile(eta_edges_pos, phi_edges)
        trackTileNeg = EtaPhiTile(eta_edges_neg, phi_edges)
        tracksterTilePos = EtaPhiTile(eta_edges_pos, phi_edges)
        tracksterTileNeg = EtaPhiTile(eta_edges_neg, phi_edges)

        # Fill track tiles
        for idx, (eta, phi) in enumerate(zip(trks[ev]["track_hgcal_eta"], trks[ev]["track_hgcal_phi"])):
            if eta > 0:
                trackTilePos.fill(eta, phi, idx)
            else:
                trackTileNeg.fill(eta, phi, idx)

        # Fill trackster tiles
        for idx, (eta, phi) in enumerate(zip(all_tracksters[ev]["barycenter_eta"], all_tracksters[ev]["barycenter_phi"])):
            if eta > 0:
                tracksterTilePos.fill(eta, phi, idx)
            else:
                tracksterTileNeg.fill(eta, phi, idx)

        # Extract track and trackster features
        track_eta   = trks[ev]["track_hgcal_eta"]
        track_phi   = trks[ev]["track_hgcal_phi"]
        track_z     = trks[ev]["track_hgcal_z"]
        track_pt    = trks[ev]["track_pt"]
        track_hits  = trks[ev]["track_missing_outer_hits"]
        track_qual  = trks[ev]["track_quality"]
        track_id    = trks[ev]["track_id"]
        trackster_z = all_tracksters[ev]["barycenter_z"]

        for track_idx, (eta, phi) in enumerate(zip(track_eta, track_phi)):
            # Selection
            #print(" track_idx ", track_idx, " all_valid_track_indices[ev] ", all_valid_track_indices[ev])
            if track_idx not in all_valid_track_indices[ev]:
                continue
            if track_hits[track_idx] > 4:
                continue
            if track_pt[track_idx] <= 1.0:
                continue
            if track_qual[track_idx] < 1:
                continue
            #print("track_z[track_idx] ", track_z[track_idx])

            node = Node(index=track_idx, is_trackster=False)

            if eta > 0:
                eta_min = max(eta - delta, min_eta_pos)
                eta_max = min(eta + delta, max_eta_pos)
                tile = tracksterTilePos
            else:
                eta_min = max(eta - delta, min_eta_neg)
                eta_max = min(eta + delta, max_eta_neg)
                tile = tracksterTileNeg

            phi_min = phi - delta
            phi_max = phi + delta

            eta_bin_min = np.digitize([eta_min], tile.eta_bins)[0]
            eta_bin_max = np.digitize([eta_max], tile.eta_bins)[0]
            phi_bin_min = np.digitize([phi_min], tile.phi_bins)[0]
            phi_bin_max = np.digitize([phi_max], tile.phi_bins)[0]

            if phi_bin_min > phi_bin_max:
                phi_bin_max += len(tile.phi_bins)

            for eta_i in range(eta_bin_min, eta_bin_max + 1):
                for phi_i in range(phi_bin_min, phi_bin_max + 1):
                    wrapped_phi_i = phi_i % len(tile.phi_bins)
                    bin_key = (eta_i, wrapped_phi_i)
                    for ts_idx in tile.tile.get(bin_key, []):
                        if np.sign(trackster_z[ts_idx]) == np.sign(track_z[track_idx]):
                            node.neighbours.append(ts_idx)
                            #print("trackster_z[ts_idx] ", trackster_z[ts_idx])

            event_nodes.append(node)
            #print("event_nodes ", event_nodes)
        all_events_nodes.append(event_nodes)

    return all_events_nodes
