#%%

import matplotlib.pyplot as plt
import wandb
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
plt.style.use('default')

#%%


def load_sweep(sweep_str, cache_dir="sweep_cache", redownload=False):
    """
    Loads sweep data from cache if available, otherwise fetches from wandb and caches it.
    Only includes finished runs in the dataframe.
    Returns a pandas DataFrame.
    """
    cache_path = os.path.join(cache_dir, f"{sweep_str.replace('/', '_')}.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path) and not redownload:
        df = pd.read_pickle(cache_path)
    else:
        api = wandb.Api()
        runs = api.sweep(sweep_str).runs
        summary_list, config_list, name_list = [], [], []

        total_runs = len(runs)
        finished_runs = 0

        print(f"Found {total_runs} total runs in sweep")

        for run in tqdm(runs):
            # Check if the run is finished before adding it to the dataframe
            if run.state == "finished":
                # print(f"Adding finished run: {run.entity}/{run.project}/{run.id}")
                summary_list.append(run.summary._json_dict)
                config = {k: v for k, v in run.config.items()} | {'run_path': f'{run.entity}/{run.project}/{run.id}', 'sweep': run.sweepName}
                config_list.append(config)
                finished_runs += 1
            else:
                print(f"Skipping {run.state} run: {run.entity}/{run.project}/{run.id}")

        print(f"Added {finished_runs} finished runs out of {total_runs} total runs")

        if finished_runs == 0:
            print("Warning: No finished runs found in sweep!")
            return pd.DataFrame()

        config_df = pd.DataFrame(config_list)
        summary_df = pd.DataFrame(summary_list)
        df = pd.concat([config_df, summary_df], axis=1)
        df = df.drop(columns=['denoiser'], errors='ignore')
        df.to_pickle(cache_path)
    return df


def load_run(run_path):
    """
    Loads a single wandb run and returns its config and summary as a pandas DataFrame.
    Only includes the run if it is finished.
    """
    api = wandb.Api()
    entity, project, run_id = run_path.split('/')
    run = api.run(f"{entity}/{project}/{run_id}")

    if run.state != "finished":
        print(f"Run {run_path} is not finished (state: {run.state}).")
        return pd.DataFrame()

    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items()} | {
        'run_path': run_path,
        'sweep': run.sweepName if hasattr(run, 'sweepName') else None
    } | {'run_path': f'{run.entity}/{run.project}/{run.id}', 'sweep': run.sweepName}
    df = pd.DataFrame([{**config, **summary}])
    # df = df.drop(columns=['denoiser'], errors='ignore')
    return df


def load_distances(run):
    """
    Loads distance data for a run. Checks if local file exists with matching size.
    If size matches, uses local file. If not, raises an error.
    """
    # Get run info
    import time
    start_time = time.time()
    
    run = wandb.Api().run(run)
    run_id = run.id
    # print(run_id)
    remote_file_str = f'outputs/{run_id}.npz'
    local_path = f'./wandb/outputs/{run_id}.npz'
    
    # Find the output file and get its size
    remote_file = run.file(remote_file_str)
    
    
    if not remote_file:
        raise FileNotFoundError(f"No output.npz file found for run {run_id}")
    # print(remote_file)
    end_time = time.time()
    # print(f"Time to load run and find output file: {end_time - start_time:.2f} seconds")
    
    # Check if local file exists with matching size
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        # if local_size == remote_file.size:
            # print(f"Using cached file for {run_id} ({local_size} bytes)")
        dist = np.load(local_path, allow_pickle=True)
        return {key: dist[key] for key in dist.keys()}
        # else:
        #     raise ValueError(f"Size mismatch: local {local_size} vs remote {remote_file.size} bytes")
    
    # Download if file doesn't exist
    # print(f"Downloading file for {run_id} ({remote_file.size} bytes)")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    remote_file.download(root="wandb/", replace=True)
    
    # Load and return the data
    dist = np.load(local_path, allow_pickle=True)
    return {key: dist[key] for key in dist.keys()}

# luwinkler/bioemu-steering-tests/szh87ilz
# luwinkler/bioemu-steering-tests/5eajf31b
# luwinkler/bioemu-steering-tests/bosihyak

# Load the sweep data
sweep_id = "luwinkler/bioemu-steering-tests/bosihyak"
sweep_df = load_sweep(sweep_id)
print(f"Initial sweep dataframe shape: {sweep_df.shape}")

# Process all runs and store distances in dictionary
distances_dict = {}
for idx, row in tqdm(sweep_df.iterrows(), total=len(sweep_df)):
    run_path = row['run_path']
    run_id = run_path.split('/')[-1]

    dist_data = load_distances(run_path)
    distances_dict[run_path] = dist_data

#%%

# for key, val in distances_dict.items():
#     dict_ = {key: val.shape for key,val in val.items()}
#     print(f"{key}: {dict_}")


# Filter sweep_df for sequence_length = 449
filtered_df = sweep_df[(sweep_df['sequence_length'] == 449) & (sweep_df['steering.potentials.caclash.dist'] == 1) & (sweep_df['steering.resample_every_n_steps'] == 3)].copy()
# print(f"Filtered dataframe shape (sequence_length=449): {filtered_df.shape}")

# Calculate violations for each run
violations_data = []
for idx, row in filtered_df.iterrows():
    run_path = row['run_path']
    if run_path in distances_dict:
        ca_ca_distances = distances_dict[run_path]['ca_ca']
        raw_violations = np.sum(ca_ca_distances > 4.5, axis=-1)
        print(raw_violations.shape)
        
        # Calculate violations with different error tolerances
        violations_data.append({
            'run_path': run_path,
            'num_particles': row['steering.num_particles'],
            'start': row['steering.start'],
            'num_violations_tol0': np.sum(np.maximum(0, raw_violations - 0)==0)/len(raw_violations),
            'num_violations_tol1': np.sum(np.maximum(0, raw_violations - 1)==0)/len(raw_violations),
            'num_violations_tol2': np.sum(np.maximum(0, raw_violations - 2)==0)/len(raw_violations),
            'num_violations_tol3': np.sum(np.maximum(0, raw_violations - 3)==0)/len(raw_violations),
            'num_violations_tol4': np.sum(np.maximum(0, raw_violations - 4)==0)/len(raw_violations),
            'num_violations_tol5': np.sum(np.maximum(0, raw_violations - 5)==0)/len(raw_violations)
        })

# Create violations dataframe
violations_df = pd.DataFrame(violations_data)
print(f"Violations dataframe shape: {violations_df.shape}")

# Create the plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))

# Separate data by steering.start value
unique_starts = sorted(violations_df['start'].unique())
base_colors = ['red', 'blue']

# Define tolerance levels (0 = no tolerance, 1-5 = with tolerance)
tolerance_levels = [0, 1, 2, 3, 4, 5]
alphas = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]  # Full opacity for tol=0, decreasing for higher tolerances

for i, start_val in enumerate(unique_starts):
    subset = violations_df[violations_df['start'] == start_val]
    base_color = base_colors[i % len(base_colors)]
    
    for j, tol in enumerate(tolerance_levels):
        # Use the tolerance column data
        y_values = subset[f'num_violations_tol{tol}']
        label = f'start={start_val}, tol={tol}'
        
        # Use circle for tolerance=0, x for tolerance>0
        marker = 'o' if tol == 0 else 'x'
        
        ax.scatter(subset['num_particles'], y_values, 
                  color=base_color, alpha=alphas[j], 
                  label=label, s=30, marker=marker)

ax.set_xlabel('steering.num_particles')
ax.set_ylabel('Fraction of structures with zero violations')
ax.set_title('Violations vs Number of Particles by Tolerance Level (sequence_length=449)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.ylim(0,1)

plt.tight_layout()
plt.show()
