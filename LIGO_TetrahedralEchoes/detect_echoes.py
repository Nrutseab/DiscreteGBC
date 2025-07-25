import numpy as np
import matplotlib.pyplot as plt
from physics.gravitational_waves import detect_discrete_waves

def main(event_file, detector='L1', output_dir="results"):
    # Detect tetrahedral echoes
    match, params, template, strain, t = detect_discrete_waves(event_file, detector)
    mass, defect = params
    
    print(f"Event: {event_file}")
    print(f"Best match: {match:.2%}")
    print(f"Parameters: mass={mass:.2f} M☉, defect={defect:.4f} rad")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, strain, 'b-', alpha=0.7, label="LIGO Strain")
    plt.plot(t, template, 'r-', linewidth=2, label="Tetrahedral Template")
    plt.title(f"GW Event {event_file} - Match: {match:.2%}")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/{event_file}_match.png")
    plt.close()
    
    return match, params

if __name__ == "__main__":
    event = "GW230529"
    match, params = main(f"data/ligo/{event}.hdf5")
