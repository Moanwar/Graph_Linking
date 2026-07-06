import os
import os.path as osp
from glob import glob

import tqdm as tqdm
from itertools import chain

import uproot as uproot
import awkward as ak
import numpy as np
from sklearn.neighbors import KDTree

import torch
from torch_geometric.data import Dataset, Data
from label_utils import assign_trackster_labels
from nodes_construction import construct_nodes
from graph_construction import construct_graphs
from min_max_dist import adding_addFeatures

class ClusterDataset(Dataset):
    feature_trkst = ["barycenter_eta","barycenter_phi","barycenter_etaError","barycenter_phiError","barycenter_x","barycenter_y","barycenter_z","raw_energy","time","timeError","raw_em_energy","raw_em_pt","raw_pt"]     
    feature_trk = ["track_hgcal_eta","track_hgcal_phi","track_hgcal_etaErr","track_hgcal_phiErr","track_hgcal_x","track_hgcal_y","track_hgcal_z","track_p","track_pt", "track_nhits"]
    model_feature_keys= np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    def __init__(self, root, histo_path, transform=None, test=False, pre_transform=None, pre_filter=None):
        self.test = test
        self.histo_path = histo_path
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return glob(f"{self.raw_dir}/*")

    @property
    def processed_file_names(self):
        return glob(f"{self.processed_dir}/data_*.pt")

    # use this to load the tree if some of file.keys() are duplicates ending with different numbers
    def load_branch_with_highest_cycle(self, file, branch_name):

        # Get all keys in the file
        all_keys = file.keys()

        # Filter keys that match the specified branch name
        matching_keys = [
            key for key in all_keys if key.startswith(branch_name)]

        if not matching_keys:
            raise ValueError(
                f"No branch with name '{branch_name}' found in the file.")

        # Find the key with the highest cycle
        highest_cycle_key = max(
            matching_keys, key=lambda key: int(key.split(";")[1]))

        # Load the branch with the highest cycle
        branch = file[highest_cycle_key]

        return branch

    def download(self):
        if (self.test):
            files = glob(f"{self.histo_path}/*.root")
        else:
            files = glob(f"{self.histo_path}/*.root")
            print("working")
        for id in range(len(files)):
            file = uproot.open(files[id])
            alltracksters = self.load_branch_with_highest_cycle(file,'ticlDumper/ticlTracksterLinks')
            allclusters = self.load_branch_with_highest_cycle(file,'ticlDumper/clusters')
            allsimtrackstersCP = self.load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersCP')
            allsimtrackstersSC = self.load_branch_with_highest_cycle(file, 'ticlDumper/simtrackstersSC')
            allassociations = self.load_branch_with_highest_cycle(file, 'ticlDumper/associations')
            alltracks = self.load_branch_with_highest_cycle(file, 'ticlDumper/tracks')

            node_feature_keys_before_trkst = ["barycenter_eta","barycenter_phi","barycenter_etaError",
                                              "barycenter_phiError","barycenter_x","barycenter_y",
                                              "barycenter_z","raw_energy","time","timeError",
                                              "raw_em_energy","raw_em_pt","raw_pt"]
            node_feature_keys_before_trk = ["track_hgcal_eta","track_hgcal_phi","track_hgcal_etaErr",
                                            "track_hgcal_phiErr","track_hgcal_x","track_hgcal_y",
                                            "track_hgcal_z","track_p","track_pt","track_nhits"]

            data_trkst = alltracksters.arrays(node_feature_keys_before_trkst)
            data_trk = alltracks.arrays(node_feature_keys_before_trk)

            simTracksters = allsimtrackstersSC.arrays(['raw_em_energy','raw_energy', 'regressed_energy',
                                                       'pdgID', 'NTracksters','NClusters'])
            tsCP=allsimtrackstersCP.arrays(["trackIdx","regressed_energy","regressed_pt"])
            all_tracksters = alltracksters.arrays(["barycenter_phi","barycenter_eta","raw_energy",
                                                   "time", "timeError","barycenter_etaError",
                                                   "barycenter_phiError", 'barycenter_x', 'barycenter_y',
                                                   'barycenter_z','vertices_x','vertices_y','vertices_z',
                                                   'vertices_indexes','raw_em_pt', 'raw_pt'])
            associations = allassociations.arrays(['ticlTracksterLinks_recoToSim_CP_sharedE',
                                                   "ticlTracksterLinks_recoToSim_CP",
                                                   "ticlTracksterLinks_recoToSim_CP_score",
                                                   "ticlTracksterLinks_simToReco_CP_score",
                                                   "ticlTracksterLinks_simToReco_CP",
                                                   "ticlTracksterLinks_simToReco_CP_sharedE"])
            trks=alltracks.arrays(["track_hgcal_eta","track_hgcal_phi","track_pt","track_id",
                                   'track_hgcal_etaErr','track_hgcal_phiErr','track_hgcal_etaphiCov',
                                   'track_p','track_beta', 'track_quality',
                                   'track_missing_outer_hits','track_nhits', 'track_time_quality',
                                   'track_time','track_missing_inner_hits', 'track_hgcal_pt',
                                   'track_hgcal_xyCov', 'track_hgcal_yErr','track_hgcal_xErr',
                                   'track_hgcal_z', 'track_hgcal_y', 'track_hgcal_x'])

            all_valid_track_indices = []
            for ev in range(len(tsCP)):
                valid_indices = []
                for i in range(len(tsCP[ev]["trackIdx"])):
                    if tsCP[ev]["trackIdx"][i] == -1:
                        continue
                    matches = np.where(trks[ev]["track_id"] == tsCP[ev]["trackIdx"][i])[0]
                    if len(matches) == 0:
                        continue
                    valid_indices.append(matches[0])
                #print(valid_indices)
                all_valid_track_indices.append(valid_indices)

            #labels_dict = assign_trackster_labels(associations, all_tracksters)
            labels_dict = assign_trackster_labels(associations, all_tracksters, simTracksters["pdgID"])
            data_trkst["y"]        = labels_dict["y"]
            data_trkst["score"]    = labels_dict["score"]
            data_trkst["shared_e"] = labels_dict["shared_e"]
            data_trkst["vertices"] = ak.concatenate([all_tracksters["vertices_x"][:, :, :, np.newaxis], all_tracksters["vertices_y"][:, :, :, np.newaxis], all_tracksters["vertices_z"][:, :, :, np.newaxis]], axis=-1)
            #data_trkst["vertices"] = np.stack([all_tracksters["vertices_x"], all_tracksters["vertices_y"], all_tracksters["vertices_z"]],axis=-1)
            data_trkst["barycenter_eta"] = np.abs(data_trkst["barycenter_eta"])
            data_trkst["barycenter_z"] = np.abs(data_trkst["barycenter_z"])            
            data_trk["track_hgcal_eta"] = np.abs(data_trk["track_hgcal_eta"])
            data_trk["track_hgcal_z"] = np.abs(data_trk["track_hgcal_z"])
            all_nodes = construct_nodes(trks, all_tracksters, all_valid_track_indices, tsCP)
            torch.save(all_nodes, osp.join(self.raw_dir, f'all_nodes_id_{id}.pt'))
            torch.save(data_trkst, osp.join(self.raw_dir, f'data_trkst_id_{id}.pt'))
            torch.save(data_trk, osp.join(self.raw_dir, f'data_trk_id_{id}.pt'))

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            if "all_nodes" not in raw_path :
                continue 
            print(f"Loading: {raw_path}")
            # Extract ID from filename (e.g. 'data_trkst_id_5.pt')
            file_id = int(os.path.basename(raw_path).split('_')[-1].split('.')[0])

            # Load associated trk and trkst files
            trk_path = osp.join(self.raw_dir, f'data_trk_id_{file_id}.pt')
            trkst_path = osp.join(self.raw_dir, f'data_trkst_id_{file_id}.pt')
            nodes_path = osp.join(self.raw_dir, f'all_nodes_id_{file_id}.pt')
            if not osp.exists(trk_path) or not osp.exists(trkst_path) or not osp.exists(nodes_path):
                print(f"Missing files for ID {file_id}, skipping.")
                continue

            data_trk = torch.load(trk_path, weights_only=False)
            data_trkst = torch.load(trkst_path, weights_only=False)
            all_nodes = torch.load(nodes_path, weights_only=False)
            graph_list = []
            #print(" len(all_nodes) ", len(all_nodes))
            #print(" len(data_trkst[y] ", len(data_trkst["y"]))
            for event in range(len(all_nodes)):
                # Reuse your own graph construction logic here
                #print(all_nodes[event])
                intialgraph = construct_graphs(event, data_trk, data_trkst, all_nodes[event],self.feature_trkst, self.feature_trk)
                graph = adding_addFeatures(event, intialgraph, data_trkst, all_nodes[event],self.feature_trk ,self.feature_trkst)
                if graph is None or graph.x.size(0) == 0 or graph.edge_index.size(1) == 0:
                    continue
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                    #print(" graph ", graph)
                torch.save(graph, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        
    def len(self):
        return len(self.processed_file_names)
    '''
    def get(self, idx):
        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        #data = torch.load(osp.join(self.processed_dir, f'data_*_{idx}.pt'), weights_only=False)
        pattern = osp.join(self.processed_dir, f"data_*_{idx}.pt")
        matches = glob(pattern)
        print(" matches ", matches)
        if len(matches) == 0:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        
        data_list = [torch.load(f, weights_only=False) for f in matches]        
        if len(data_list) > 1:
            from torch_geometric.data import Batch
            data = Batch.from_data_list(data_list)
        else:
            data = data_list[0]
        print(data)
        return data
    '''
    def get(self, idx):
        pattern = osp.join(self.processed_dir, f"data_*_{idx}.pt")
        matches = glob(pattern)
        print("matches:", matches)
    
        if len(matches) == 0:
            print(f"[Warning] Skipping missing file for pattern: {pattern}")
            return None  # Skip missing samples
        
        data_list = [torch.load(f, weights_only=False) for f in matches]

        if len(data_list) > 1:
            from torch_geometric.data import Batch
            data = Batch.from_data_list(data_list)
        else:
            data = data_list[0]

        return data
