#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:30:10 2026

@author: uranus
"""

import numpy as np
import pandas as pd
import mne

# ------------------ LOAD ACCURACIES ------------------
accuracy_face = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_faces_final.xlsx')
accuracy_scr = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_scr_final.xlsx')
accuracy_bodies = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_bodies_final.xlsx')
accuracy_tool = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_tools_final.xlsx')
accuracy_scene = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_scenes_final.xlsx')

# ------------------ STORAGE ------------------
templates = {
    "face": [],
    "body": [],
    "tool": [],
    "scene": [],
    "scr": []
}

peak_times = {
    "face": [],
    "body": [],
    "tool": [],
    "scene": [],
    "scr": []
}
acc_idx = 0
# ------------------ LOOP ------------------
for k in range(1, 43):

    if k in [10, 41, 29]:
        continue

    print(f"\nProcessing P{k}")

    # ------------------ LOAD EEG ------------------
    fname = f'/media/uranus/Elements/Tangerine_preprocessed/preprocessed_data/TASK_FINAL/VERSION2/P{k}_Task_Final_2.set'
    task_data = mne.io.read_epochs_eeglab(fname)

    # remove eye channels (NEW API)
    bads = ['TIME','L-GAZE-X','L-GAZE-Y','L-AREA','R-GAZE-X','R-GAZE-Y','R-AREA','INPUT']
    task_data = task_data.pick([ch for ch in task_data.ch_names if ch not in bads])

    task_data.set_montage('GSN-HydroCel-256')
   # task_data = task_data.resample(sfreq=100)
    times = task_data.times  # 500 Hz EEG time

    # ------------------ LOAD BEHAVIOR ------------------
    df = pd.read_excel(f'/media/uranus/Elements/behavioural/Nuova cartella/P{k}.xlsx')

    conds = {
        "face": df[(df['Image']=='faces') & (df['Block'].isin([1,2,3,4]))].index,
        "body": df[(df['Image']=='bodies') & (df['Block'].isin([1,2,3,4]))].index,
        "tool": df[(df['Image']=='tools') & (df['Block'].isin([1,2,3,4]))].index,
        "scene": df[(df['Image']=='scenes') & (df['Block'].isin([1,2,3,4]))].index,
        "scr": df[(df['Image']=='scrambled') & (df['Block'].isin([1,2,3,4]))].index
    }

    evokeds = {name: task_data[idx].average() for name, idx in conds.items()}

    # ------------------ LOAD ACCURACY ------------------
    accs = {
        "face": accuracy_face.iloc[acc_idx].values,
        "body": accuracy_bodies.iloc[acc_idx].values,
        "tool": accuracy_tool.iloc[acc_idx].values,
        "scene": accuracy_scene.iloc[acc_idx].values,
        "scr": accuracy_scr.iloc[acc_idx].values
    }
    acc_idx += 1  
    # ------------------ ACCURACY TIME (100 Hz) ------------------
     
    max_time = 0.499  # seconds
    task_data_resample = task_data.resample(sfreq=100)
    time_indices = np.where(task_data_resample.times <= max_time)[0]
    acc_times= task_data_resample.times[time_indices]
    # ------------------ LOOP CONDITIONS ------------------
    for name, acc in accs.items():

        # ---- YOUR STYLE: mask in accuracy space ----
        mask = (acc_times >= 0.05) & (acc_times < 0.25)        
        times_before = acc_times[mask]
        values_before = acc[mask]

        max_acc = np.max(values_before)
        first_idx = np.where(values_before == max_acc)[0][0]
        t_peak = times_before[first_idx]

        # ---- MAP TO EEG (500 Hz) ----
        idx_eeg = np.argmin(np.abs(task_data.times - t_peak))        # ---- OPTIONAL: small window (recommended) ----
       
        topo = evokeds[name].data[:, idx_eeg]

        # ---- STORE ----
        templates[name].append(topo)
        peak_times[name].append(t_peak)

        print(f"{name}: {t_peak:.3f}s")

# ------------------ GRAND AVERAGE ------------------
grand_templates = {}
for name in templates:
    grand_templates[name] = np.mean(np.stack(templates[name]), axis=0)

# ------------------ SAVE ------------------
np.save('/media/uranus/Elements/ANALYSIS/templates.npy', templates)
np.save('/media/uranus/Elements/ANALYSIS/peak_times.npy', peak_times)
np.save('/media/uranus/Elements/ANALYSIS/grand_templates.npy', grand_templates)

print("\n✅ DONE — templates + peak times saved")




##############################


import matplotlib.pyplot as plt
import mne

for name in templates:

    n_subj = len(templates[name])
    n_cols = 6
    n_rows = int(np.ceil(n_subj / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()

    for i, topo in enumerate(templates[name]):
        mne.viz.plot_topomap(
            topo,
            task_data.info,
            axes=axes[i],
            show=False
        )
        axes[i].set_title(f"S{i+1}")

    # remove empty plots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{name} – individual topomaps")
    plt.tight_layout()
    plt.show()
    
    
    
 for name in peak_times:

    plt.figure()

    plt.hist(peak_times[name], bins=10)

    plt.xlabel("Peak time (s)")
    plt.ylabel("Count")
    plt.title(f"{name} – peak time distribution")

    plt.axvline(0, linestyle='--')      # stimulus onset
    plt.axvline(0.25, linestyle='--')   # your cutoff

    plt.show()