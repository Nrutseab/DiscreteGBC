import h5py
import numpy as np
from physics.gravitational_waves import tetrahedral_waveform

def create_ligo_event(filename, mass=2.6, defect=0.14, noise_level=0.2, seed=None):
    """Generate sample LIGO event data with tetrahedral signal"""
    if seed is not None:
        np.random.seed(seed)
        
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
        "GW150914": {'mass': 35.0, 'defect': 0.08, 'seed': None},
        "GW170817": {'mass': 1.4, 'defect': 0.12, 'seed': None},
        "GW230529": {'mass': 2.6, 'defect': 0.14, 'seed': 123},  # Fixed seed for reproducibility
        "GW190814": {'mass': 23.0, 'defect': 0.09, 'seed': None}
    }
    
    for event, params in events.items():
        filename = f"data/ligo/{event}.hdf5"
        print(f"Generating {event} with params: {params}")
        create_ligo_event(filename, **params)

if __name__ == "__main__":
    generate_all_events()
