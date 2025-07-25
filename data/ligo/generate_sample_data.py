import h5py
import numpy as np
from physics.gravitational_waves import tetrahedral_waveform

def create_ligo_event(filename, mass=2.6, defect=0.14, noise_level=0.2):
    """Generate sample LIGO event data with tetrahedral signal"""
    sampling_rate = 4096  # Hz
    duration = 0.5  # seconds
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Generate signal
    _, signal = tetrahedral_waveform(t, mass, defect, sampling_rate)
    
    # Add noise
    noise = noise_level * np.random.normal(size=len(t))
    strain = signal + noise
    
    # Create HDF5 file
    with h5py.File(filename, 'w') as f:
        grp = f.create_group('L1')
        grp.create_dataset('strain', data=strain)
        grp.create_dataset('t0', data=0.0)
        grp.create_dataset('sampling_rate', data=sampling_rate)
        
        # Store parameters for verification
        grp.attrs['mass'] = mass
        grp.attrs['defect'] = defect

def generate_all_events():
    events = {
        "GW150914": (35.0, 0.08),   # First BH-BH detection
        "GW170817": (1.4, 0.12),     # Neutron star merger
        "GW230529": (2.6, 0.14),     # Recent mass-gap event
        "GW190814": (23.0, 0.09)     # Mystery object event
    }
    
    for event, params in events.items():
        filename = f"data/ligo/{event}.hdf5"
        print(f"Generating {event} with mass={params[0]} M☉, defect={params[1]} rad")
        create_ligo_event(filename, mass=params[0], defect=params[1])

if __name__ == "__main__":
    generate_all_events(
        
    )
