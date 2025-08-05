import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the building details and transmitter positions"""
    # Load building details
    building_details = pd.read_csv('B24_Details.csv')
    print("Building Details:")
    print(building_details)
    
    # Load transmitter positions
    positions = pd.read_csv('Positions_B24_Ant1_f1.csv')
    print(f"\nTransmitter Positions Shape: {positions.shape}")
    print("First few positions:")
    print(positions.head())
    
    return building_details, positions

def perform_kmeans_clustering(positions, k=3):
    """Perform K-means clustering on transmitter positions"""
    # Extract X and Y coordinates
    coordinates = positions[['X', 'Y']].values
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)
    
    # Get cluster centers (selected transmitter locations)
    cluster_centers = kmeans.cluster_centers_
    
    # Find the closest actual transmitter to each cluster center
    selected_indices = []
    selected_positions = []
    
    for center in cluster_centers:
        # Calculate distances from center to all transmitters
        distances = np.sqrt(np.sum((coordinates - center)**2, axis=1))
        closest_idx = np.argmin(distances)
        selected_indices.append(closest_idx)
        selected_positions.append(positions.iloc[closest_idx])
    
    return cluster_labels, cluster_centers, selected_indices, selected_positions

def create_visualization(building_details, positions, cluster_labels, cluster_centers, selected_indices):
    """Create and save the visualization as PDF"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Get selected positions
    selected_positions = positions.iloc[selected_indices]
    
    # Define ripple parameters
    num_ripples = 6  # Number of concentric circles per transmitter
    max_radius = 50  # Maximum radius of the outermost ripple
    ripple_color = 'lightgray'  # Very light gray for all ripples
    
    # Add ripple effects for each selected transmitter
    for i, (idx, row) in enumerate(selected_positions.iterrows()):
        tx_x, tx_y = row['X'], row['Y']
        
        # Create multiple concentric circles with decreasing alpha (ripple effect)
        for ripple in range(num_ripples):
            radius = max_radius * (ripple + 1) / num_ripples
            alpha = 0.6 * (num_ripples - ripple) / num_ripples  # Stronger at center, weaker outward
            
            # Create circle
            circle = plt.Circle((tx_x, tx_y), radius, 
                              color=ripple_color, 
                              fill=False, 
                              linewidth=2.5 - (ripple * 0.3),  # Thicker lines for inner circles
                              alpha=alpha)
            ax.add_patch(circle)
            
            # Add filled circle with very low alpha for gradient effect
            filled_circle = plt.Circle((tx_x, tx_y), radius, 
                                     color=ripple_color, 
                                     fill=True, 
                                     alpha=alpha * 0.1)
            ax.add_patch(filled_circle)
    
    # Plot all candidate transmitter locations
    scatter = ax.scatter(positions['X'], positions['Y'], 
                        c='lightblue', 
                        marker='^', 
                        s=100, 
                        alpha=0.9,
                        edgecolors='darkblue',
                        linewidth=1,
                        label='Candidate Tx Locations',
                        zorder=5)  # Higher z-order to appear above ripples
    
    # Plot selected transmitter locations
    ax.scatter(selected_positions['X'], selected_positions['Y'], 
              c='red', 
              marker='^', 
              s=180, 
              alpha=1.0,
              edgecolors='darkred',
              linewidth=2,
              label='Selected Tx Locations',
              zorder=10)  # Highest z-order to appear on top
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=24)
    ax.set_ylabel('Y', fontsize=24)
    # ax.set_title('K-means Selection of Transmitter Locations with Signal Diffusion (K=3)', fontsize=18, fontweight='bold')
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add legend with increased font size
    ax.legend(loc='upper right', fontsize=24)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # # Add text box with statistics
    # textstr = f'Total Candidates: {len(positions)}\nSelected Tx: 3'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=22,
    #         verticalalignment='top', bbox=props, zorder=15)
    
    # Adjust plot limits to accommodate ripples
    x_min, x_max = positions['X'].min(), positions['X'].max()
    y_min, y_max = positions['Y'].min(), positions['Y'].max()
    margin = max_radius + 10
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages('transmitter_selection_kmeans.pdf') as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        print("Plot saved as 'transmitter_selection_kmeans.pdf'")
    
    plt.show()
    
    return fig

def print_results(selected_positions, cluster_centers):
    """Print the results of the selection"""
    print("\n" + "="*50)
    print("K-MEANS TRANSMITTER SELECTION RESULTS")
    print("="*50)
    
    print(f"\nSelected Transmitter Locations:")
    for i, pos in enumerate(selected_positions):
        print(f"Transmitter {i+1}:")
        print(f"  Position: ({pos['X']}, {pos['Y']})")
        if 'Azimuth' in pos:
            print(f"  Azimuth: {pos['Azimuth']}°")
        print(f"  Signal Diffusion: Multi-layer ripples with decreasing intensity")
        print()

def main():
    """Main function to execute the transmitter selection process"""
    print("Starting K-means Transmitter Selection Analysis...")
    print("-" * 50)
    
    # Load data
    building_details, positions = load_data()
    
    # Perform K-means clustering
    cluster_labels, cluster_centers, selected_indices, selected_positions = perform_kmeans_clustering(positions, k=3)
    
    # Create visualization
    fig = create_visualization(building_details, positions, cluster_labels, cluster_centers, selected_indices)
    
    # Print results
    print_results(selected_positions, cluster_centers)
    
    print("Analysis completed successfully!")
    
    return selected_positions, cluster_centers

if __name__ == "__main__":
    selected_transmitters, centers = main()