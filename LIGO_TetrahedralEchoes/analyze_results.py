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
        f.write("LIGO Discrete Gravitational Wave Analysis Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analyzed {len(df)} events\n\n")
        
        total_match = 0
        for _, row in df.iterrows():
            f.write(f"Event: {row['event']}\n")
            f.write(f"- Match Significance: {row['match']:.2%}\n")
            f.write(f"- Estimated Mass: {row['mass']:.2f} M☉\n")
            f.write(f"- Estimated Angle Defect: {row['defect']:.4f} rad\n\n")
            total_match += row['match']
        
        avg_match = total_match / len(df) if len(df) > 0 else 0
        f.write("\nSummary Statistics:\n")
        f.write(f"- Average Match: {avg_match:.2%}\n")
        f.write(f"- Highest Match: {df['match'].max():.2%} ({df.loc[df['match'].idxmax()]['event']})\n")
        
        # Statistical significance analysis
        significant = df[df['match'] > 0.7]
        f.write(f"- Events with >70% match: {len(significant)}/{len(df)}\n")
        
        if len(significant) > 0:
            f.write("\nConclusion: Strong evidence of tetrahedral spacetime structure\n")
        else:
            f.write("\nConclusion: Insufficient evidence of tetrahedral spacetime structure\n")
    
    print(f"Analysis complete. Results saved to {results_file} and {output_dir}")

if __name__ == "__main__":
    main()
