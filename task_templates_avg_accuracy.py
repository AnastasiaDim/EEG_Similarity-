#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:30:10 2026

@author: uranus
"""

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
import mne
# LOAD ACCURACIES ------------------
accuracy_face = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_faces_final.xlsx')
accuracy_scr = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_scr_final.xlsx')
accuracy_bodies = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_bodies_final.xlsx')
accuracy_tool = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_tools_final.xlsx')
accuracy_scene = pd.read_excel('/media/uranus/Elements/ANALYSIS/Replay_paper/classification_results/results/accuracy_scores_scenes_final.xlsx')

avg_face = accuracy_face.mean(axis=0)
avg_body = accuracy_bodies.mean(axis=0)
avg_tool = accuracy_tool.mean(axis=0)
avg_scene = accuracy_scene.mean(axis=0)
avg_scr = accuracy_scr.mean(axis=0)
# Find peak times per category

times = task_data.resample(sfreq=100).times 
max_time = 0.499  # seconds
task_data_resample = task_data.resample(sfreq=100)
time_indices = np.where(task_data_resample.times <= max_time)[0]
acc_times= task_data_resample.times[time_indices] # 500 Hz EEG time

face_time  = acc_times[np.argmax(avg_face)]
body_time  = acc_times[np.argmax(avg_body)]
tool_time  = acc_times[np.argmax(avg_tool)]
scene_time = acc_times[np.argmax(avg_scene)]
mask_early = acc_times < 0.25

# Apply the mask to both time and accuracy
scr_times_early = acc_times[mask_early]
scr_acc_early   = avg_scr[mask_early]

# Find the index of the maximum in the masked array
idx_peak = np.argmax(scr_acc_early)

# Convert index back to time
scr_time = scr_times_early[idx_peak]

print(f"Scrambled peak before 0.25s: {scr_time:.3f}s")
# Store in a dictionary for convenience
mean_peaks = {
    "face": face_time,
    "body": body_time,
    "tool": tool_time,
    "scene": scene_time,
    "scr": scr_time
}

# Print for verification
for name, t_peak in mean_peaks.items():
    print(f"{name}: mean peak at {t_peak:.3f} s")




fig, ax = plt.subplots(figsize=(8,4))

# Plot accuracy
ax.plot(acc_times, avg_face, label='Accuracy')

# Shaded region from 0.14 to 0.16 s
ax.axvspan(0.12, 0.18, color='green', alpha=0.3, label='Window ±10 ms')

# Labels
ax.set_title("Classification Accuracy - First Participant")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Accuracy")
ax.legend()

plt.tight_layout()
plt.show()


window_size = 0.0  





templates = {name: [] for name in mean_peaks.keys()}

for k in range(1, 44):

    if k in [10, 29, 41]:
        continue

    print(f"\nProcessing P{k}")

    # ------------------ LOAD EEG ------------------
    fname = f'/media/uranus/Elements/Tangerine_preprocessed/preprocessed_data/TASK_FINAL/VERSION2/P{k}_Task_Final_2.set'
    task_data = mne.io.read_epochs_eeglab(fname)

    bads = ['TIME','L-GAZE-X','L-GAZE-Y','L-AREA','R-GAZE-X','R-GAZE-Y','R-AREA','INPUT']
    task_data = task_data.pick([ch for ch in task_data.ch_names if ch not in bads])
    task_data.set_montage('GSN-HydroCel-256')

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

    # ------------------ EXTRACT TOPOGRAPHIES AT FIXED TIMES ------------------
    for name, t_peak in mean_peaks.items():

        idx_eeg = np.argmin(np.abs(task_data.times - t_peak))

        #win_mask = (task_data.times >= t_peak - window_size) & (task_data.times <= t_peak + window_size)
        #topo_window = np.mean(evokeds[name].data[:, win_mask], axis=1)
        topo = evokeds[name].data[:, idx_eeg]

        # Store
        templates[name].append(topo)

        print(f"{name}: extracted EEG at {t_peak:.3f}s for P{k}")
        
        
        
grand_templates = {}
for name, topo_list in templates.items():
    grand_templates[name] = np.mean(np.stack(topo_list), axis=0)
    
    


for name in templates.keys():
    n_participants = len(templates[name])
    n_cols = 6  # participants per row
    n_rows = math.ceil(n_participants / n_cols) + 1  # +1 row for grand-average

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()  # flatten for easy indexing

    # Individual topomaps
    for i, topo in enumerate(templates[name]):
        mne.viz.plot_topomap(topo, evokeds[name].info, axes=axes[i], show=False)
        axes[i].set_title(f"P{i+1}", fontsize=10)

    # Fill remaining subplots with blank if needed
    for j in range(i+1, n_rows*n_cols-1):
        axes[j].axis('off')

    # Grand-average in the last subplot
    mne.viz.plot_topomap(grand_templates[name], evokeds[name].info, axes=axes[-1], show=False)
    axes[-1].set_title(f"Grand Average", fontsize=12)

    plt.suptitle(f"Topographies - {name}", fontsize=16)
    plt.tight_layout()
    plt.show()  
    


categories = ["face", "body", "tool", "scene", "scr"]

fig, axes = plt.subplots(1, len(categories), figsize=(15,4))

for i, cat in enumerate(categories):
    mne.viz.plot_topomap(grand_templates[cat], evokeds[cat].info,
                         axes=axes[i], show=False)
    axes[i].set_title(f"{cat} - Grand Avg")

plt.suptitle("Grand-Average Topographies per Category", fontsize=16)
plt.tight_layout()
plt.show()   
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

categories = ["face", "body", "tool", "scene", "scr"]

# Store individual similarity matrices
participant_category_similarity = []

for p_idx in range(len(templates["face"])):  # loop over participants
    # Stack topographies for this participant (5 categories × channels)
    topographies = np.array([templates[cat][p_idx] for cat in categories])
    
    # Compute cosine similarity: result is 5x5
    sim_matrix = cosine_similarity(topographies)
    participant_category_similarity.append(sim_matrix)

# Convert to numpy array: participants × 5 × 5
participant_category_similarity = np.stack(participant_category_similarity)


# Average over participants
grand_category_similarity = np.mean(participant_category_similarity, axis=0)

print("Grand-average category similarity matrix:\n", np.round(grand_category_similarity, 3))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(grand_category_similarity, vmin=0, vmax=1, cmap='coolwarm')

# Labels
ax.set_xticks(range(len(categories)))
ax.set_yticks(range(len(categories)))
ax.set_xticklabels(categories)
ax.set_yticklabels(categories)
plt.title("Grand-Average Category Similarity (Cosine)")

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Cosine similarity")

plt.tight_layout()
plt.show()


save_path = "/media/uranus/Elements/ANALYSIS/Study_2/similarity/avg_timepint/templates.npz"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

templates_array = {k: np.stack(v) for k, v in templates.items()}

np.savez(save_path, **templates_array)

print("Saved templates to:", save_path)

data = np.load("/media/uranus/Elements/ANALYSIS/Study_2/similarity/avg_timepint/templates.npz")

templates = {k: data[k] for k in data.files}

import pickle

with open("/media/uranus/Elements/ANALYSIS/Study_2/similarity/avg_timepint/mean_peaks.pkl", "wb") as f:
    pickle.dump(mean_peaks, f)
    
    

with open("/media/uranus/Elements/ANALYSIS/Study_2/similarity/avg_timepint/mean_peaks.pkl", "rb") as f:
    mean_peaks = pickle.load(f)

print(mean_peaks)


