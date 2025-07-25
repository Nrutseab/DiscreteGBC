import numpy as np
from scipy.spatial import Delaunay

def compute_dihedral_angle(points):
    """Compute dihedral angles for a tetrahedron"""
    normals = []
    for i in range(4):
        face = np.delete(points, i, axis=0)
        vec1 = face[1] - face[0]
        vec2 = face[2] - face[0]
        normal = np.cross(vec1, vec2)
        normals.append(normal / np.linalg.norm(normal))
    
    angles = []
    for i in range(4):
        for j in range(i+1, 4):
            cos_angle = np.dot(normals[i], normals[j])
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)
            
    return angles

def angle_defect(complex, hinge):
    """Compute angle defect at a hinge"""
    total_angle = 0
    for tet in complex.tets_containing_hinge(hinge):
        points = complex.get_points(tet)
        angles = compute_dihedral_angle(points)
        total_angle += angles[complex.hinge_index(hinge, tet)]
        
    return 2 * np.pi - total_angle

def discrete_gbc_integral(complex):
    """Compute the Gauss-Bonnet-Chern integral"""
    total = 0
    for hinge in complex.hinges:
        defect = angle_defect(complex, hinge)
        dual_vol = complex.dual_volume(hinge)
        total += defect * dual_vol
    return total
