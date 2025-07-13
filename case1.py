import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class Case1PromptSelector:
    
    def __init__(self, layout_shape=(400, 350), random_state=42):
        self.layout_shape = layout_shape
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def generate_feasible_tx_positions(self, n_positions=1000, exclude_walls=True):
        width, height = self.layout_shape
        
        x_coords = np.random.uniform(0, width, n_positions)
        y_coords = np.random.uniform(0, height, n_positions)
        
        positions = np.column_stack((x_coords, y_coords))
        
        if exclude_walls:
            margin = 20  
            valid_mask = (
                (positions[:, 0] > margin) & (positions[:, 0] < width - margin) &
                (positions[:, 1] > margin) & (positions[:, 1] < height - margin)
            )
            positions = positions[valid_mask]
        
        return positions
    
    def select_representative_tx_positions(self, candidate_positions, k=5, max_iterations=100):
        kmeans = KMeans(
            n_clusters=k, 
            random_state=self.random_state, 
            max_iter=max_iterations,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(candidate_positions)
        centroids = kmeans.cluster_centers_
        
        selected_positions = []
        cluster_info = {
            'centroids': centroids,
            'cluster_labels': cluster_labels,
            'clusters': {},
            'selected_indices': []
        }
        
        for cluster_idx in range(k):
            cluster_mask = cluster_labels == cluster_idx
            cluster_points = candidate_positions[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_points) > 0:
                centroid = centroids[cluster_idx]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                closest_idx_in_cluster = np.argmin(distances)
                closest_global_idx = cluster_indices[closest_idx_in_cluster]
                
                selected_position = candidate_positions[closest_global_idx]
                selected_positions.append(selected_position)
                
                cluster_info['clusters'][cluster_idx] = {
                    'points': cluster_points,
                    'centroid': centroid,
                    'selected_position': selected_position,
                    'selected_index': closest_global_idx
                }
                cluster_info['selected_indices'].append(closest_global_idx)
        
        return np.array(selected_positions), cluster_info
    
    def simulate_ray_tracing(self, tx_positions):
        prompts = []
        
        for i, tx_pos in enumerate(tx_positions):
            input_tensor = self._create_input_tensor(tx_pos)
            pathloss_map = self._simulate_pathloss(tx_pos)
            prompts.append({
                'input': input_tensor,
                'output': pathloss_map,
                'tx_position': tx_pos,
                'prompt_id': i
            })
        
        return prompts
    
    def _create_input_tensor(self, tx_pos):
        """Create 3-channel input tensor (distance, transmittance, reflectance)."""
        height, width = self.layout_shape[1], self.layout_shape[0]
        
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        distance_map = np.sqrt((x_coords - tx_pos[0])**2 + (y_coords - tx_pos[1])**2)
        
        
        input_tensor = np.stack([distance_map, transmittance_map, reflectance_map], axis=2)
        return input_tensor
    
    def _simulate_pathloss(self, tx_pos):
        """Simulate pathloss map using simplified propagation model."""
        height, width = self.layout_shape[1], self.layout_shape[0]
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        distances = np.sqrt((x_coords - tx_pos[0])**2 + (y_coords - tx_pos[1])**2)
        pl0 = 30  
        n = 2.5  
        d0 = 1.0  
        distances = np.maximum(distances, d0)
        pathloss = pl0 + 10 * n * np.log10(distances / d0)
        
        
        return pathloss
    
    def visualize_selection(self, candidate_positions, selected_positions, cluster_info):
        """Visualize the K-means clustering and selected positions."""
        plt.figure(figsize=(12, 8))
        
        plt.scatter(candidate_positions[:, 0], candidate_positions[:, 1], 
                   c=cluster_info['cluster_labels'], alpha=0.6, s=20, cmap='tab10')
        
        centroids = cluster_info['centroids']
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.scatter(selected_positions[:, 0], selected_positions[:, 1], 
                   c='red', marker='*', s=300, edgecolors='black', linewidths=2, 
                   label='Selected Tx Positions')
        
        plt.xlim(0, self.layout_shape[0])
        plt.ylim(0, self.layout_shape[1])
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('K-means Clustering for Transmitter Position Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    selector = Case1PromptSelector(layout_shape=(400, 350))
    
    print("Generating candidate transmitter positions...")
    candidate_positions = selector.generate_feasible_tx_positions(n_positions=500)
    print(f"Generated {len(candidate_positions)} feasible positions")
    
    k = 5  
    print(f"\nSelecting {k} representative positions using K-means clustering...")
    selected_positions, cluster_info = selector.select_representative_tx_positions(
        candidate_positions, k=k
    )
    
    print(f"Selected {len(selected_positions)} representative positions:")
    for i, pos in enumerate(selected_positions):
        print(f"  Position {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    print("\nSimulating ray-tracing for selected positions...")
    prompts = selector.simulate_ray_tracing(selected_positions)
    
    print(f"Generated {len(prompts)} prompt examples")
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}: Input shape {prompt['input'].shape}, "
              f"Output shape {prompt['output'].shape}")
    
    print("\nVisualizing selection process...")
    selector.visualize_selection(candidate_positions, selected_positions, cluster_info)