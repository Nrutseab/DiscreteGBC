import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from physics.gravitational_waves import detect_discrete_waves

def main(results_file="results/ligo_analysis.csv", output_dir="results/plots"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load events
    events = ["GW150914", "GW170817", "GW230529", "GW190814"]
    results = []
    
    print("Analyzing LIGO events...")
    for event in events:
        print(f"Processing {event}...")
        match, params, _, _, _ = detect_discrete_waves(f"data/ligo/{event}.hdf5")
        results.append({
            "event": event,
            "match": match,
            "mass": params[0],
            "defect": params[1]
        })
        
        # Save plot
        plt.figure(figsize=(10, 6))
        plt.title(f"{event} - Match: {match:.2%}")
        plt.savefig(f"{output_dir}/{event}_match.png")
        plt.close()
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    
    # Generate summary report
    with open(f"{output_dir}/summary_report.txt", "w") as f:
        f.write("LIGO Discrete Gravitational Wave Analysis Report\\n")
        f.write("="*50 + "\\n\\n")
        
        for _, row in df.iterrows():
            f.write(f"Event: {row['event']}\\n")
            f.write(f"- Match Significance: {row['match']:.2%}\\n")
            f.write(f"- Estimated Mass: {row['mass']:.2f
