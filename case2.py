import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import random
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class Case2PromptRetriever:
    def __init__(self, embedding_dim=512, random_state=42):
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.encoder = None
        self.database = []
        self.database_embeddings = None
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)
    
    def _create_scene_encoder(self):
        encoder = models.resnet18(pretrained=True)
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                                 padding=3, bias=False)
        encoder.fc = nn.Linear(encoder.fc.in_features, self.embedding_dim)
        encoder.eval()
        return encoder
    
    def initialize_encoder(self):
        self.encoder = self._create_scene_encoder()
        print(f"Initialized scene encoder with {self.embedding_dim}D embeddings")
    
    def create_database(self, n_layouts=100, layouts_per_type=20):
        database = []
        layout_types = ['office', 'residential', 'warehouse', 'hospital', 'retail']
        for layout_idx in range(n_layouts):
            layout_type = layout_types[layout_idx % len(layout_types)]
            input_tensor, output_map = self._generate_synthetic_example(
                layout_type, layout_idx
            )
            database.append({
                'input': input_tensor,
                'output': output_map,
                'layout_type': layout_type,
                'layout_id': layout_idx,
                'tx_position': np.random.uniform(50, 350, 2)  # Random Tx position
            })
        self.database = database
        print(f"Created database with {len(database)} examples")
        return database
    
    def compute_scene_embeddings(self, batch_size=BS):
        if self.encoder is None:
            self.initialize_encoder()
        embeddings = []
        n_samples = len(self.database)
        print(f"Computing embeddings for {n_samples} database entries...")
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_inputs = []
            for i in range(start_idx, end_idx):
                input_tensor = self.database[i]['input']
                input_normalized = self._preprocess_input(input_tensor)
                batch_inputs.append(input_normalized)
            batch_tensor = torch.stack(batch_inputs)
            with torch.no_grad():
                batch_embeddings = self.encoder(batch_tensor)
                embeddings.append(batch_embeddings.numpy())
        self.database_embeddings = np.vstack(embeddings)
        print(f"Computed embeddings shape: {self.database_embeddings.shape}")
        return self.database_embeddings
    
    def _preprocess_input(self, input_tensor):
        normalized = np.zeros_like(input_tensor)
        for c in range(input_tensor.shape[2]):
            channel = input_tensor[:, :, c]
            normalized[:, :, c] = (channel - channel.mean()) / (channel.std() + 1e-8)
        tensor = torch.FloatTensor(normalized).permute(2, 0, 1)
        return tensor
    
    def retrieve_top_k_prompts(self, query_input, k=5):
        if self.database_embeddings is None:
            print("Computing database embeddings...")
            self.compute_scene_embeddings()
        query_preprocessed = self._preprocess_input(query_input)
        query_batch = query_preprocessed.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            query_embedding = self.encoder(query_batch).numpy()
        similarities = cosine_similarity(query_embedding, self.database_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1] 
        top_k_similarities = similarities[top_k_indices]
        selected_prompts = []
        for idx in top_k_indices:
            prompt = self.database[idx].copy()
            prompt['similarity'] = similarities[idx]
            prompt['database_index'] = idx
            selected_prompts.append(prompt)
        print(f"Retrieved top-{k} prompts with similarities: {top_k_similarities}")
        return selected_prompts, top_k_similarities
    
    def visualize_retrieval(self, query_input, selected_prompts, similarities):
        k = len(selected_prompts)
        fig, axes = plt.subplots(2, k+1, figsize=(4*(k+1), 8))
        axes[0, 0].imshow(query_input[:, :, 0], cmap='viridis')  
        axes[0, 0].set_title('Query\n(Distance)')
        axes[0, 0].axis('off')
        axes[1, 0].imshow(query_input[:, :, 1], cmap='plasma')   
        axes[1, 0].set_title('Query\n(Transmittance)')
        axes[1, 0].axis('off')
        for i, (prompt, sim) in enumerate(zip(selected_prompts, similarities)):
            col = i + 1
            axes[0, col].imshow(prompt['input'][:, :, 0], cmap='viridis')
            axes[0, col].set_title(f'Prompt {i+1}\nSim: {sim:.3f}\n({prompt["layout_type"]})')
            axes[0, col].axis('off')
            axes[1, col].imshow(prompt['output'], cmap='hot')
            axes[1, col].set_title(f'Pathloss Map {i+1}')
            axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_retrieval_quality(self, query_input, selected_prompts):
        print("\n=== Retrieval Analysis ===")
        layout_types = [p['layout_type'] for p in selected_prompts]
        similarities = [p['similarity'] for p in selected_prompts]
        print(f"Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
        query_stats = self._compute_input_statistics(query_input)
        for i, prompt in enumerate(selected_prompts):
            stats = self._compute_input_statistics(prompt['input'])
            print(f"  Prompt {i+1} ({prompt['layout_type']}):")
            print(f" Distance range: {stats['distance_range']}")
            print(f" Transmittance mean: {stats['transmittance_mean']:.2f}")
            print(f" Reflectance mean: {stats['reflectance_mean']:.2f}")

    def _compute_input_statistics(self, input_tensor):
        return {
            'distance_range': (input_tensor[:, :, 0].min(), input_tensor[:, :, 0].max()),
            'transmittance_mean': input_tensor[:, :, 1].mean(),
            'reflectance_mean': input_tensor[:, :, 2].mean()
        }

if __name__ == "__main__":
    retriever = Case2PromptRetriever(embedding_dim=256)
    database = retriever.create_database(n_layouts=20)
    retriever.initialize_encoder()
    embeddings = retriever.compute_scene_embeddings(batch_size=32)
    k = 3
    print(f"\nRetrieving top-{k} similar prompts...")
    selected_prompts, similarities = retriever.retrieve_top_k_prompts(query_input, k=k)
    retriever.analyze_retrieval_quality(query_input, selected_prompts)
    print("\nVisualizing retrieval results...")
    retriever.visualize_retrieval(query_input, selected_prompts, similarities)
    print(f"\nSummary:")
    print(f"- Retrieved {len(selected_prompts)} most similar prompts")
    print(f"- Best similarity score: {max(similarities):.3f}")
