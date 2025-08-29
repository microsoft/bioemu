import matplotlib.pyplot as plt
import os
import pandas as pd
import wandb
from tqdm import tqdm

plt.style.use('default')
def load_sweep(sweep_str, cache_dir="sweep_cache"):
    """
    Loads sweep data from cache if available, otherwise fetches from wandb and caches it.
    Returns a pandas DataFrame.
    """
    cache_path = os.path.join(cache_dir, f"{sweep_str.replace('/', '_')}.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
    else:
        api = wandb.Api()
        runs = api.sweep(sweep_str).runs
        summary_list, config_list, name_list = [], [], []

        for run in tqdm(runs):
            if run.state == "finished":
                summary_list.append(run.summary._json_dict)
                config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
                name_list.append(run.name)

        config_df = pd.DataFrame(config_list)
        summary_df = pd.DataFrame(summary_list)
        df = pd.concat([config_df, summary_df], axis=1)
        df = df.drop(columns=['denoiser'], errors='ignore')
        df.to_pickle(cache_path)
    return df


# Example usage:
df = load_sweep("luwinkler/bioemu-steering-tests/1w7zls3f")
df_baseline = load_sweep("luwinkler/bioemu-steering-tests/uvneuaxv")

# Identify unique values in config columns (sweep parameters)
print("Unique values for steering columns:")
print("=" * 50)

for col in df.columns:
    if col.startswith('steering.'):
        unique_vals = df[col].dropna().unique()
        print(f"\n{col}:")
        for val in sorted(unique_vals):
            print(f"  - {val}")

# Get unique sequence lengths
seq_lengths = [60, 449]

# Create subplot for each sequence length

for idx, seq_len in enumerate(seq_lengths):
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    ax = axes
    ax2 = ax.twinx()  # Create second y-axis
    
    # Filter data for this sequence length
    df_seq = df[df['sequence_length'] == seq_len]
    
    # Get unique steering.start values
    start_vals = sorted(df_seq['steering.start'].dropna().unique())
    
    for start_val in start_vals:
        # Filter for this start value
        df_start = df_seq[df_seq['steering.start'] == start_val]
        
        # Group by number of particles and get mean physicality metrics
        grouped_clash = df_start.groupby('steering.num_particles')['Physicality/ca_clash<3.4 [#]'].mean()
        grouped_break = df_start.groupby('steering.num_particles')['Physicality/ca_break>4.5 [#]'].mean()
        
        # Assign different markers based on start value
        if start_val == 0.5:
            marker_clash = 's'  # square
            marker_break = 's'
        elif start_val == 0.9:
            marker_clash = '^'  # triangle
            marker_break = '^'
        else:
            marker_clash = 'o'  # circle (default)
            marker_break = 'o'

        # Add baseline horizontal lines for this sequence length
        df_baseline_seq = df_baseline[df_baseline['sequence_length'] == seq_len]
        if not df_baseline_seq.empty:
            baseline_clash = df_baseline_seq['Physicality/ca_clash<3.4 [#]'].mean()
            baseline_break = df_baseline_seq['Physicality/ca_break>4.5 [#]'].mean()
            
            ax.axhline(y=baseline_clash, color='blue', linestyle='--', alpha=0.7, 
                      label='baseline clash')
            ax.axhline(y=baseline_break, color='red', linestyle='--', alpha=0.7,
                       label='baseline break')
        
        # Plot with solid/dashed line based on start value
        linestyle = '-' if start_val == min(start_vals) else '--'
        ax.plot(grouped_clash.index, grouped_clash.values, 'b-', 
                label=f'clash start={start_val}', marker=marker_clash)
        ax.plot(grouped_break.index, grouped_break.values, 'r-',
                 label=f'break start={start_val}', marker=marker_break)
    
    # Color the axes
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylabel('CA Clash < 3.4 [#]', color='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('CA Break > 4.5 [#]', color='red')
    
    ax.set_xlabel('Number of Particles')
    ax.set_title(f'Sequence Length {seq_len}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
