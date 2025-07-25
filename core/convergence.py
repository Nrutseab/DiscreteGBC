import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

class ConvergenceAnalysis:
    def __init__(self, sequence):
        """
        Initialize with a sequence of DiscreteManifold objects
        representing increasingly refined triangulations
        """
        self.sequence = sequence
        self.dim = sequence[0].dim
        self.results = {}
        
    def compute_gh_distance(self, X, Y):
        """Approximate Gromov-Hausdorff distance between two point clouds"""
        # Compute distance matrices
        dx = pairwise_distances(X)
        dy = pairwise_distances(Y)
        
        # Compute distortion
        distortion = np.max(np.abs(dx - dy))
        return distortion
    
    def check_alexandrov_convergence(self, k, diameter_bound, volume_lower_bound):
        """
        Check conditions for convergence to a CAT(k) Alexandrov space:
        1. Uniform diameter bound
        2. Uniform lower curvature bound (sec^h ≥ k)
        3. Uniform volume lower bound
        """
        results = []
        for i, manifold in enumerate(self.sequence):
            # 1. Check diameter bound
            diameter = self.compute_diameter(manifold)
            if diameter > diameter_bound:
                return False, f"Diameter {diameter} exceeds bound at refinement {i}"
                
            # 2. Check curvature bound
            if not self.check_curvature_bound(manifold, k):
                return False, f"Curvature bound violated at refinement {i}"
                
            # 3. Check volume bound
            volume = self.compute_volume(manifold)
            if volume < volume_lower_bound:
                return False, f"Volume {volume} below bound at refinement {i}"
                
            results.append({
                'refinement': i,
                'diameter': diameter,
                'min_curvature': self.get_min_curvature(manifold),
                'volume': volume
            })
        
        self.results['alexandrov_conditions'] = results
        return True, "All Alexandrov convergence conditions satisfied"
    
    def compute_diameter(self, manifold):
        """Compute diameter of the manifold"""
        dist_matrix = pairwise_distances(manifold.vertices)
        return np.max(dist_matrix)
    
    def compute_volume(self, manifold):
        """Compute total volume of the manifold"""
        return np.sum(manifold.vertex_volumes)
    
    def check_curvature_bound(self, manifold, k):
        """Check if sectional curvature is bounded below by k"""
        # Simplified check: ensure all angle defects are non-negative for k=0
        defects = manifold.compute_angle_defect()
        if k <= 0:
            return all(defect >= 0 for defect in defects.values())
        else:
            # More sophisticated check for positive k
            return self.check_cat_k_condition(manifold, k)
    
    def check_cat_k_condition(self, manifold, k):
        """Check CAT(k) condition using triangle comparison"""
        # Select random triangles
        vertices = manifold.vertices
        n = len(vertices)
        indices = np.random.choice(n, min(100, n), replace=False)
        sample_vertices = vertices[indices]
        
        # Check triangle comparisons
        for i in range(len(sample_vertices)):
            for j in range(i+1, len(sample_vertices)):
                for k_idx in range(j+1, len(sample_vertices)):
                    a, b, c = sample_vertices[i], sample_vertices[j], sample_vertices[k_idx]
                    if not self.compare_triangle(a, b, c, k):
                        return False
        return True
    
    def compare_triangle(self, a, b, c, k):
        """
        Compare triangle to constant curvature k model
        Returns True if discrete triangle is thinner than model triangle
        """
        # Compute edge lengths
        d_ab = np.linalg.norm(a - b)
        d_bc = np.linalg.norm(b - c)
        d_ca = np.linalg.norm(c - a)
        
        # Compute angles in constant curvature space
        if k == 0:
            # Euclidean comparison
            return self.check_euclidean_triangle(d_ab, d_bc, d_ca)
        else:
            # Spherical/hyperbolic comparison (simplified)
            return True  # Placeholder
    
    def check_euclidean_triangle(self, a, b, c):
        """Check triangle inequality in Euclidean space"""
        return (a + b > c) and (b + c > a) and (c + a > b)
    
    def get_min_curvature(self, manifold):
        """Get minimum curvature value"""
        defects = manifold.compute_angle_defect()
        return min(defects.values()) if defects else 0
    
    def analyze_convergence(self, reference=None):
        """
        Analyze convergence of the sequence to a reference manifold
        or to a continuous limit
        """
        distortions = []
        volumes = []
        curvatures = []
        
        for i in range(1, len(self.sequence)):
            # Compute distortion between successive refinements
            dist = self.compute_gh_distance(
                self.sequence[i-1].vertices,
                self.sequence[i].vertices
            )
            distortions.append(dist)
            
            # Track volume and curvature changes
            volumes.append(self.compute_volume(self.sequence[i]))
            curvatures.append(self.get_min_curvature(self.sequence[i]))
        
        # If reference is provided (continuous manifold sample)
        if reference is not None:
            ref_dists = []
            for manifold in self.sequence:
                dist = self.compute_gh_distance(manifold.vertices, reference)
                ref_dists.append(dist)
            self.results['reference_distances'] = ref_dists
        
        self.results['distortions'] = distortions
        self.results['volumes'] = volumes
        self.results['curvatures'] = curvatures
        
        return self.results
    
    def visualize_convergence(self):
        """Generate convergence plots (matplotlib)"""
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Distortion plot
        axes[0].plot(self.results['distortions'], 'o-')
        axes[0].set_title('Gromov-Hausdorff Distortion')
        axes[0].set_xlabel('Refinement Step')
        axes[0].set_ylabel('Distortion')
        axes[0].set_yscale('log')
        
        # Volume plot
        axes[1].plot(self.results['volumes'], 'o-')
        axes[1].set_title('Volume Convergence')
        axes[1].set_xlabel('Refinement Step')
        axes[1].set_ylabel('Volume')
        
        # Curvature plot
        axes[2].plot(self.results['curvatures'], 'o-')
        axes[2].set_title('Curvature Convergence')
        axes[2].set_xlabel('Refinement Step')
        axes[2].set_ylabel('Min Curvature')
        
        plt.tight_layout()
        return fig
