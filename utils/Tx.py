import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

def load_data():
    building_details = pd.read_csv('.../B24_Details.csv')
    print("Building Details:")
    print(building_details)
    positions = pd.read_csv('.../Positions_B24_Ant1_f1.csv')
    print(f"\nTx Positions Shape: {positions.shape}")
    print(positions.head())
    return building_details, positions

def perform_kmeans_clustering(positions, k=3):
    coordinates = positions[['X', 'Y']].values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)
    cluster_centers = kmeans.cluster_centers_
    selected_indices = []
    selected_positions = []
    for center in cluster_centers:
        distances = np.sqrt(np.sum((coordinates - center)**2, axis=1))
        closest_idx = np.argmin(distances)
        selected_indices.append(closest_idx)
        selected_positions.append(positions.iloc[closest_idx])
    
    return cluster_labels, cluster_centers, selected_indices, selected_positions

def create_visualization(building_details, positions, cluster_labels, cluster_centers, selected_indices):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    selected_positions = positions.iloc[selected_indices]
    max_radius = 256 
    ripple_color = 'lightgray'  # Very light gray for all ripples
    for i, (idx, row) in enumerate(selected_positions.iterrows()):
        tx_x, tx_y = row['X'], row['Y']
        for ripple in range(num_ripples):
            radius = max_radius * (ripple + 1) / num_ripples
            alpha = 0.6 * (num_ripples - ripple) / num_ripples  # Stronger at center, weaker outward
            circle = plt.Circle((tx_x, tx_y), radius, 
                              color=ripple_color, 
                              fill=False, 
                              linewidth=2.5 - (ripple * 0.3),  # Thicker lines for inner circles
                              alpha=alpha)
            ax.add_patch(circle)
            filled_circle = plt.Circle((tx_x, tx_y), radius, 
                                     color=ripple_color, 
                                     fill=True, 
                                     alpha=alpha * 0.1)
            ax.add_patch(filled_circle)
            
    scatter = ax.scatter(positions['X'], positions['Y'], 
                        c='lightblue', 
                        marker='^', 
                        s=100, 
                        alpha=0.9,
                        edgecolors='darkblue',
                        linewidth=1,
                        label='Candidate Tx Locations',
                        zorder=5)  
    
    ax.scatter(selected_positions['X'], selected_positions['Y'], 
              c='red', 
              marker='^', 
              s=180, 
              alpha=1.0,
              edgecolors='darkred',
              linewidth=2,
              label='Selected Tx Locations',
              zorder=10)  
    
    ax.set_xlabel('X', fontsize=24)
    ax.set_ylabel('Y', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='upper right', fontsize=24)
    ax.set_aspect('equal', adjustable='box')
    x_min, x_max = positions['X'].min(), positions['X'].max()
    y_min, y_max = positions['Y'].min(), positions['Y'].max()
    margin = max_radius + 10
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    with PdfPages('Tx_selection_kmeans.pdf') as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def print_results(selected_positions, cluster_centers):
    print("\n" + "="*50)
    print("="*50)
    print(f"\nSelected Transmitter Locations:")
    for i, pos in enumerate(selected_positions):
        print(f"Transmitter {i+1}:")
        print(f"  Position: ({pos['X']}, {pos['Y']})")
        if 'Azimuth' in pos:
            print(f"  Azimuth: {pos['Azimuth']}")

def main():
    print("Tx Selection Analysis...")
    print("-" * 50)
    building_details, positions = load_data()
    cluster_labels, cluster_centers, selected_indices, selected_positions = perform_kmeans_clustering(positions, k=3)
    fig = create_visualization(building_details, positions, cluster_labels, cluster_centers, selected_indices)
    print_results(selected_positions, cluster_centers)
    print("Analysis completed")
    return selected_positions, cluster_centers

if __name__ == "__main__":
    selected_transmitters, centers = main()
