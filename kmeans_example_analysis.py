#!/usr/bin/env python3
"""
K-means Clustering: Intuitive Example with Mathematical Analysis
================================================================

This script demonstrates K-means clustering with synthetic data and provides
visualizations showing algorithm convergence and cluster formation.

Author: Research Analysis
Date: July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Set style for academic plots
plt.style.use('default')
sns.set_palette("husl")

class KMeansAnalysis:
    """
    A class to demonstrate and analyze K-means clustering with mathematical insights.
    """
    
    def __init__(self, n_samples=300, n_centers=3, random_state=42):
        """
        Initialize the analysis with synthetic data.
        
        Parameters:
        -----------
        n_samples : int
            Number of data points to generate
        n_centers : int
            Number of true clusters in the data
        random_state : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_centers = n_centers
        self.random_state = random_state
        self.generate_data()
        
    def generate_data(self):
        """Generate synthetic clustered data for demonstration."""
        # Create synthetic data with known clusters
        self.X, self.y_true = make_blobs(
            n_samples=self.n_samples,
            centers=self.n_centers,
            cluster_std=1.5,
            center_box=(-10.0, 10.0),
            random_state=self.random_state
        )
        
        # Create a more complex dataset for comparison
        self.X_complex, self.y_complex = make_blobs(
            n_samples=400,
            centers=4,
            cluster_std=2.0,
            center_box=(-15.0, 15.0),
            random_state=self.random_state + 1
        )
        
    def calculate_wcss(self, X, centroids, labels):
        """
        Calculate Within-Cluster Sum of Squares (WCSS).
        
        Mathematical formula: WCSS = Σᵢ₌₁ᵏ Σₓ∈Sᵢ ||x - μᵢ||²
        """
        wcss = 0
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[i]) ** 2)
        return wcss
    
    def kmeans_step_by_step(self, X, k=3, max_iter=10):
        """
        Implement K-means step by step to show convergence.
        
        Returns:
        --------
        history : list
            List of (centroids, labels, wcss) for each iteration
        """
        # Random initialization
        np.random.seed(self.random_state)
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
        
        history = []
        
        for iteration in range(max_iter):
            # Assignment step: assign each point to nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Calculate WCSS for this iteration
            wcss = self.calculate_wcss(X, centroids, labels)
            history.append((centroids.copy(), labels.copy(), wcss))
            
            # Update step: recalculate centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=1e-4):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            centroids = new_centroids
            
        return history
    
    def plot_convergence_analysis(self):
        """Create comprehensive plots showing K-means convergence."""
        # Perform step-by-step K-means
        history = self.kmeans_step_by_step(self.X, k=3)
        
        # Create subplot layout
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Original data with true clusters
        plt.subplot(2, 3, 1)
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y_true, 
                            cmap='viridis', alpha=0.7, s=50)
        plt.title('Original Data with True Clusters', fontsize=12, fontweight='bold')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(scatter)
        
        # Plot 2-4: First three iterations of K-means
        for i, (centroids, labels, wcss) in enumerate(history[:3]):
            plt.subplot(2, 3, i + 2)
            scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, 
                                cmap='tab10', alpha=0.7, s=50)
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
            
            # Draw circles around clusters
            for j, centroid in enumerate(centroids):
                cluster_points = self.X[labels == j]
                if len(cluster_points) > 0:
                    radius = np.max(np.sqrt(((cluster_points - centroid)**2).sum(axis=1)))
                    circle = Circle(centroid, radius, fill=False, 
                                  linestyle='--', alpha=0.5, color='red')
                    plt.gca().add_patch(circle)
            
            plt.title(f'Iteration {i+1}\nWCSS = {wcss:.2f}', 
                     fontsize=12, fontweight='bold')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
        
        # Plot 5: WCSS convergence
        plt.subplot(2, 3, 5)
        wcss_values = [wcss for _, _, wcss in history]
        plt.plot(range(1, len(wcss_values) + 1), wcss_values, 
                'bo-', linewidth=2, markersize=8)
        plt.title('WCSS Convergence', fontsize=12, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.grid(True, alpha=0.3)
        
        # Add mathematical annotation
        plt.text(0.02, 0.95, r'$WCSS = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$', 
                transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        
        # Plot 6: Final clustering result
        plt.subplot(2, 3, 6)
        final_centroids, final_labels, final_wcss = history[-1]
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=final_labels, 
                            cmap='tab10', alpha=0.7, s=50)
        plt.scatter(final_centroids[:, 0], final_centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Final Centroids')
        plt.title(f'Final Clustering\nWCSS = {final_wcss:.2f}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/mayankdw/fast_delivery_paper/kmeans_convergence_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_elbow_method(self):
        """Plot the elbow method for optimal K selection."""
        k_range = range(1, 11)
        wcss_values = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(self.X)
            wcss_values.append(kmeans.inertia_)
        
        # Create elbow plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=8)
        plt.title('Elbow Method for Optimal K Selection', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.grid(True, alpha=0.3)
        
        # Highlight the elbow point
        plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Optimal K=3')
        plt.legend()
        
        # Add mathematical formula
        plt.text(0.02, 0.95, r'$J = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('/Users/mayankdw/fast_delivery_paper/kmeans_elbow_method.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_centroid_movement(self):
        """Visualize how centroids move during iterations."""
        history = self.kmeans_step_by_step(self.X, k=3, max_iter=8)
        
        plt.figure(figsize=(12, 8))
        
        # Plot data points
        plt.scatter(self.X[:, 0], self.X[:, 1], c='lightgray', alpha=0.5, s=30)
        
        # Plot centroid paths
        colors = ['red', 'blue', 'green']
        for cluster_id in range(3):
            centroid_path = np.array([centroids[cluster_id] for centroids, _, _ in history])
            
            # Plot path
            plt.plot(centroid_path[:, 0], centroid_path[:, 1], 
                    color=colors[cluster_id], linewidth=2, alpha=0.7,
                    label=f'Cluster {cluster_id + 1} Centroid Path')
            
            # Plot centroid positions
            plt.scatter(centroid_path[:, 0], centroid_path[:, 1], 
                       color=colors[cluster_id], s=100, alpha=0.8)
            
            # Annotate iterations
            for i, (x, y) in enumerate(centroid_path):
                plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor=colors[cluster_id], alpha=0.3))
        
        plt.title('Centroid Movement During K-means Iterations', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add update formula
        plt.text(0.02, 0.02, r'$\mu_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{x \in S_i^{(t)}} x$', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('/Users/mayankdw/fast_delivery_paper/kmeans_centroid_movement.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_statistics(self):
        """Generate summary statistics and mathematical analysis."""
        # Fit K-means
        kmeans = KMeans(n_clusters=3, random_state=self.random_state)
        labels = kmeans.fit_predict(self.X)
        centroids = kmeans.cluster_centers_
        
        # Create summary table
        summary_data = []
        total_wcss = 0
        
        for i in range(3):
            cluster_points = self.X[labels == i]
            cluster_size = len(cluster_points)
            
            if cluster_size > 0:
                # Calculate cluster statistics
                centroid = centroids[i]
                wcss_cluster = np.sum((cluster_points - centroid) ** 2)
                avg_distance = np.mean(np.sqrt(((cluster_points - centroid)**2).sum(axis=1)))
                max_distance = np.max(np.sqrt(((cluster_points - centroid)**2).sum(axis=1)))
                
                total_wcss += wcss_cluster
                
                summary_data.append({
                    'Cluster': f'Cluster {i+1}',
                    'Size': cluster_size,
                    'Centroid_X': f'{centroid[0]:.3f}',
                    'Centroid_Y': f'{centroid[1]:.3f}',
                    'WCSS': f'{wcss_cluster:.3f}',
                    'Avg_Distance': f'{avg_distance:.3f}',
                    'Max_Distance': f'{max_distance:.3f}'
                })
        
        # Create DataFrame
        df_summary = pd.DataFrame(summary_data)
        
        # Save to CSV
        df_summary.to_csv('/Users/mayankdw/fast_delivery_paper/kmeans_cluster_summary.csv', index=False)
        
        print("K-means Clustering Analysis Summary")
        print("=" * 50)
        print(f"Total Data Points: {len(self.X)}")
        print(f"Number of Clusters: 3")
        print(f"Total WCSS: {total_wcss:.3f}")
        print(f"Average WCSS per cluster: {total_wcss/3:.3f}")
        print("\nCluster Details:")
        print(df_summary.to_string(index=False))
        
        return df_summary

def main():
    """Main function to run the complete K-means analysis."""
    print("K-means Clustering: Mathematical Analysis and Visualization")
    print("=" * 60)
    
    # Initialize analysis
    analysis = KMeansAnalysis(n_samples=300, n_centers=3, random_state=42)
    
    # Generate all visualizations
    print("\n1. Generating convergence analysis plots...")
    analysis.plot_convergence_analysis()
    
    print("\n2. Generating elbow method plot...")
    analysis.plot_elbow_method()
    
    print("\n3. Generating centroid movement visualization...")
    analysis.plot_centroid_movement()
    
    print("\n4. Generating summary statistics...")
    summary_df = analysis.generate_summary_statistics()
    
    print(f"\nAll files saved to: /Users/mayankdw/fast_delivery_paper/")
    print("Generated files:")
    print("- kmeans_convergence_analysis.png")
    print("- kmeans_elbow_method.png") 
    print("- kmeans_centroid_movement.png")
    print("- kmeans_cluster_summary.csv")
    print("- kmeans_mathematical_formulation.md")

if __name__ == "__main__":
    main()