import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def angle_defect(simplex, dihedral_angles):
    """Compute angle defect at hinge"""
    return 2*np.pi - np.sum(dihedral_angles)

def discrete_ricci_curvature(vertex, neighbors, defects, dual_vol):
    """Compute discrete Ricci curvature at vertex"""
    hinge_defects = [defects[h] for h in neighbors]
    return np.sum(hinge_defects) / dual_vol

def discrete_scalar_curvature(vertex, neighbors, defects, dual_vol):
    """Compute discrete scalar curvature"""
    return len(neighbors) / dual_vol

def gbc_integral(complex):
    """Compute Gauss-Bonnet-Chern integral"""
    total = 0
    for hinge in complex.hinges:
        defect = angle_defect(hinge, complex.dihedral_angles[hinge])
        dual_vol = complex.dual_volume(hinge)
        total += defect * dual_vol
    return total

def euler_characteristic(complex):
    """Compute Euler characteristic"""
    V = len(complex.vertices)
    E = len(complex.edges)
    F = len(complex.faces)
    T = len(complex.tetrahedra) if hasattr(complex, 'tetrahedra') else 0
    return V - E + F - (T if T > 0 else 0)
