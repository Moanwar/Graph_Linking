# label_utils.py

import awkward as ak

def assign_trackster_labels(associations, all_tracksters, sim_pdgIDs):
    labels = []
    scores = []
    shared_energies = []

    for ev_idx in range(len(associations)):
        assEv = associations[ev_idx]
        recoToSimEv = assEv.ticlTracksterLinks_recoToSim_CP
        recoToSimScores = assEv.ticlTracksterLinks_recoToSim_CP_score
        recoToSimSharedE = assEv.ticlTracksterLinks_recoToSim_CP_sharedE

        simToRecoEv = assEv.ticlTracksterLinks_simToReco_CP
        simToRecoScores = assEv.ticlTracksterLinks_simToReco_CP_score
        simToRecoSharedE = assEv.ticlTracksterLinks_simToReco_CP_sharedE

        trackster_energies = all_tracksters.raw_energy[ev_idx]

        reco_labels = []
        reco_scores = []
        reco_sharedE = []

        for reco_idx in range(len(recoToSimEv)):
            simMatches = recoToSimEv[reco_idx]
            if len(simMatches) != 1:
                reco_labels.append(-1)
                reco_scores.append(-1)
                reco_sharedE.append(-1)
                continue

            sim_idx = simMatches[0]
            if sim_idx < 0:
                reco_labels.append(-1)
                reco_scores.append(-1)
                reco_sharedE.append(-1)
                continue

            simRecoMatches = simToRecoEv[sim_idx]
            simRecoScores = simToRecoScores[sim_idx]
            simRecoSharedE = simToRecoSharedE[sim_idx]

            trackster_energy = trackster_energies[reco_idx]
            backmatch_good = []

            for i, matched_reco in enumerate(simRecoMatches):
                if matched_reco == reco_idx:
                    if simRecoScores[i] < 0.9 and (simRecoSharedE[i] / trackster_energy) > 0.4:
                        backmatch_good.append(i)
                        
            pdgID = sim_pdgIDs[ev_idx][sim_idx]
            is_charged_hadron = abs(pdgID) not in (310, 130)
            #print(" pdgID ", pdgID, " is_charged_hadron ", is_charged_hadron)            
            if backmatch_good and recoToSimScores[reco_idx] < 0.6 and is_charged_hadron:
            #if backmatch_good and recoToSimScores[reco_idx] < 0.6:
                #print(" pdgID ", pdgID, " is_charged_hadron ", "True")
                reco_labels.append(reco_idx)
                reco_scores.append(recoToSimScores[reco_idx])
                reco_sharedE.append(recoToSimSharedE[reco_idx])
            else:
                #print(" pdgID ", pdgID, " is_charged_hadron ", "False")
                reco_labels.append(-1)
                reco_scores.append(-1)
                reco_sharedE.append(-1)

        labels.append(reco_labels)
        scores.append(reco_scores)
        shared_energies.append(reco_sharedE)

    return {
        "y": ak.Array(labels),
        "score": ak.Array(scores),
        "shared_e": ak.Array(shared_energies)
    }
