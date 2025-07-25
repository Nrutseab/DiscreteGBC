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

if __name__ == "__main__":
    create_ligo_event("data/ligo/GW230529.hdf5", mass=2.6, defect=0.14)
    create_ligo_event("data/ligo/GW150914.hdf5", mass=35.0, defect=0.08)
