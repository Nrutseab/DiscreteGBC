import unittest
import numpy as np
from core.curvature import DiscreteManifold
from core.convergence import ConvergenceAnalysis

class TestConvergence(unittest.TestCase):
    def setUp(self):
        # Create a sequence of increasingly refined meshes
        self.sequence = []
        for n in [10, 20, 40, 80]:
            vertices = np.random.rand(n, 2)
            simplices = self.triangulate(vertices)
            self.sequence.append(DiscreteManifold(vertices, simplices, dim=2))
    
    def triangulate(self, vertices):
        # Simple triangulation for 2D points
        from scipy.spatial import Delaunay
        return Delaunay(vertices).simplices
    
    def test_alexandrov_convergence(self):
        analyzer = ConvergenceAnalysis(self.sequence)
        valid, message = analyzer.check_alexandrov_convergence(
            k=0, 
            diameter_bound=5.0,
            volume_lower_bound=0.1
        )
        self.assertTrue(valid, message)
        
        # Check diameter decreases with refinement
        diameters = [r['diameter'] for r in analyzer.results['alexandrov_conditions']]
        self.assertTrue(all(diameters[i] >= diameters[i+1] for i in range(len(diameters)-1)))
    
    def test_gh_convergence(self):
        analyzer = ConvergenceAnalysis(self.sequence)
        results = analyzer.analyze_convergence()
        
        # Distortion should decrease with refinement
        distortions = results['distortions']
        self.assertTrue(all(distortions[i] >= distortions[i+1] for i in range(len(distortions)-1)))
        
        # Volume should be stable
        volumes = results['volumes']
        self.assertTrue(np.std(volumes) < 0.1 * np.mean(volumes))
    
    def test_visualization(self):
        analyzer = ConvergenceAnalysis(self.sequence)
        analyzer.analyze_convergence()
        fig = analyzer.visualize_convergence()
        fig.savefig("test_convergence_plot.png")

if __name__ == "__main__":
    unittest.main()
