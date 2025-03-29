import os
#add seed to control for hashing order
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Added for CUDA determinism
import sys
import logging
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from sklearn.preprocessing import LabelEncoder
from captum.attr import IntegratedGradients
from statsmodels.stats.multitest import multipletests

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Make PyTorch operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.info("Starting to run...\n")

############################################
# Worker initialization function for DataLoader
def seed_worker(worker_id):
    """
    Sets seeds for workers in DataLoader to ensure reproducible behavior.
    """
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

############################################
class CellDataset(Dataset):
    def __init__(self, df, gene_names):
        self.df = df
        self.gene_names = gene_names
        self.X = df[gene_names].values.astype(np.float32)
        self.y = df["Group_enc"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

############################################
class BetaVAEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes=2):
        super(BetaVAEClassifier, self).__init__()
        # Reset seed right before weight initialization for reproducible weights
        torch.manual_seed(42)
        
        # Encoder with more layers for better disentanglement
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Additional layer
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder with more layers
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc_dec2 = nn.Linear(hidden_dim // 2, hidden_dim)  # Additional layer
        self.fc_dec3 = nn.Linear(hidden_dim, input_dim)
        
        # Classifier remains the same
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        # Initialize weights using a deterministic method
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization to ensure deterministic behavior.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use a fixed initialization scheme
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))  # Additional layer
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reset seed for sampling - this is crucial for VAE reproducibility
        torch.manual_seed(42)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        h = F.relu(self.fc_dec2(h))  # Additional layer
        recon = self.fc_dec3(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits

############################################
def loss_function(recon, x, mu, logvar, logits, labels, beta=4.0):
    """
    Beta-VAE loss function with a beta parameter to control disentanglement.
    A higher beta forces the model to learn more disentangled representations.
    """
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence with beta weight
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = beta * kl_loss  # Apply beta weight
    
    # Classification loss remains the same
    class_loss = F.cross_entropy(logits, labels)
    
    # Total loss
    total_loss = recon_loss + kl_loss + class_loss
    
    # Return all loss components for monitoring
    return total_loss, recon_loss, kl_loss, class_loss

############################################
def classifier_forward(x):
    mu, logvar = model.encode(x)
    logits = model.classifier(mu)
    return logits

############################################
# Calculate disentanglement metrics
def calculate_disentanglement(model, dataloader, device):
    """
    Calculate a simple disentanglement score by measuring the variance
    of each latent dimension across the dataset.
    
    Higher variance across different dimensions suggests better disentanglement.
    """
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            mu, _ = model.encode(batch_x)
            latent_vectors.append(mu.cpu().numpy())
    
    # Concatenate all latent vectors
    latent_vectors = np.vstack(latent_vectors)
    
    # Calculate variance for each dimension
    dim_variances = np.var(latent_vectors, axis=0)
    
    # Calculate coefficient of variation for each dimension
    dim_means = np.abs(np.mean(latent_vectors, axis=0))
    dim_means = np.where(dim_means < 1e-10, 1e-10, dim_means)  # Avoid division by zero
    coef_variation = dim_variances / dim_means
    
    # Metrics to track disentanglement
    variance_ratio = np.max(dim_variances) / np.min(dim_variances)
    active_dims = np.sum(dim_variances > 0.01)
    
    return {
        'variance_per_dim': dim_variances,
        'coef_variation': coef_variation,
        'variance_ratio': variance_ratio,
        'active_dimensions': active_dims
    }

############################################
batch_size = 128
num_epochs = 30  # Increased epochs for better disentanglement
learning_rate = 1e-3
latent_dim = 32
hidden_dim = 256  # Increased for better representation capacity
beta = 4.0  # Beta parameter for the β-VAE (adjust as needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
adata_file = '/orcd/home/002/yongheng/PNF/PNF_GEO_complete.h5ad'
base_output_dir = '/orcd/home/002/yongheng/PNF/BetaVAE_Results_20250326_Yongheng_v1'

# Create the directory if it doesn't exist
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir, exist_ok=True)
    log.info(f"Created output directory: {base_output_dir}")
else:
    log.info(f"Output directory already exists: {base_output_dir}")

adata = sc.read(adata_file)
log.info(f"Shape of the dataset: {adata.shape}\n")

sc.pp.log1p(adata)
sc.pp.scale(adata)

# Sort cell types for consistent processing order
cell_types = sorted(adata.obs["Cell_type"].unique())

for cell_type in cell_types:
    log.info(f"Processing cell type: {cell_type}\n")

    cell_type_dir = os.path.join(base_output_dir, f"{cell_type}")

    os.makedirs(cell_type_dir, exist_ok=True)
    adata_subset = adata[adata.obs["Cell_type"] == cell_type].copy()
    expr_array = adata_subset.X.A if hasattr(adata_subset.X, "A") else adata_subset.X
    expr_df = pd.DataFrame(expr_array, index=adata_subset.obs_names, columns=adata_subset.var_names)

    expr_df["Patient"] = adata_subset.obs["Patient"].values
    expr_df["Group"] = expr_df["Patient"].apply(lambda x: "IGF" if x.startswith("IGF") else ("PNF" if x.startswith("PNF") else "Other"))
    expr_df = expr_df[expr_df["Group"].isin(["IGF", "PNF"])].copy()

    # Sort dataframe for deterministic order
    expr_df = expr_df.sort_index()

    if "Other" in expr_df["Group"].unique():
        logging.error("Unknown group present in the dataset!!\n")
        sys.exit()
        
    # Ensure consistent label encoding
    sorted_groups = sorted(expr_df["Group"].unique())
    le = LabelEncoder()
    le.fit(sorted_groups)
    expr_df["Group_enc"] = le.transform(expr_df["Group"])

    gene_names = adata_subset.var_names.tolist()
    input_dim = len(gene_names)

    dataset = CellDataset(expr_df, gene_names)
    #added a deterministic random number generator for DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=g,
        worker_init_fn=seed_worker
    )

    # Initialize the Beta-VAE model
    model = BetaVAEClassifier(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_model_state = None
    model.train()

    # For tracking loss components
    all_losses = {
        'total': [],
        'recon': [],
        'kl': [],
        'class': []
    }

    for epoch in range(num_epochs):
        # Reset seed at the beginning of each epoch with a unique but deterministic value
        epoch_seed = 42 + epoch
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)
        
        total_loss = 0
        recon_losses = 0
        kl_losses = 0
        class_losses = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            # Set a unique but deterministic seed for each batch
            batch_seed = 42 + epoch * 1000 + batch_idx
            torch.manual_seed(batch_seed)
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar, logits = model(batch_x)
            loss, r_loss, k_loss, c_loss = loss_function(
                recon, batch_x, mu, logvar, logits, batch_y, beta=beta
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            recon_losses += r_loss.item() * batch_x.size(0)
            kl_losses += k_loss.item() * batch_x.size(0)
            class_losses += c_loss.item() * batch_x.size(0)
        
        avg_loss = total_loss / len(dataset)
        avg_recon_loss = recon_losses / len(dataset)
        avg_kl_loss = kl_losses / len(dataset)
        avg_class_loss = class_losses / len(dataset)
        
        # Store losses for plotting
        all_losses['total'].append(avg_loss)
        all_losses['recon'].append(avg_recon_loss)
        all_losses['kl'].append(avg_kl_loss)
        all_losses['class'].append(avg_class_loss)
        
        log.info(f"Epoch {epoch+1}/{num_epochs} for cell type {cell_type}, "
                 f"Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
                 f"KL: {avg_kl_loss:.4f}, Class: {avg_class_loss:.4f}")

        # We're looking for >1% improvement
        if avg_loss < best_loss * 0.99:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info(f"Early stopping triggered for cell type {cell_type} at epoch {epoch+1}\n")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Calculate disentanglement metrics
    disent_metrics = calculate_disentanglement(model, dataloader, device)
    
    # Save disentanglement metrics
    pd.DataFrame({
        'dimension': range(latent_dim),
        'variance': disent_metrics['variance_per_dim'],
        'coef_variation': disent_metrics['coef_variation']
    }).to_csv(os.path.join(cell_type_dir, "disentanglement_metrics.csv"), index=False)
    
    log.info(f"Disentanglement metrics for {cell_type}:")
    log.info(f"  Variance ratio: {disent_metrics['variance_ratio']:.4f}")
    log.info(f"  Active dimensions: {disent_metrics['active_dimensions']} / {latent_dim}")

    # Save loss history
    pd.DataFrame(all_losses).to_csv(os.path.join(cell_type_dir, "training_losses.csv"), index=False)

    # Reset seed for evaluation
    torch.manual_seed(42)  
    model.eval()
    
    ig = IntegratedGradients(classifier_forward)
    # Creating a reference point for IG algorithm 
    baseline = torch.zeros((1, input_dim)).to(device)

    attributions_group = {cls: [] for cls in le.classes_}
    details_group = {cls: [] for cls in le.classes_}

    all_X = torch.tensor(dataset.X, device=device)
    all_y = torch.tensor(dataset.y, device=device)
    
    # Extract and save latent representations
    latent_vectors = []
    labels = []
    patient_ids = []
    
    with torch.no_grad():
        for i in range(len(all_X)):
            # Set a unique but deterministic seed for each sample
            sample_seed = 42 + i
            torch.manual_seed(sample_seed)
            
            cell_id = dataset.df.index[i]
            patient = dataset.df.loc[cell_id, "Patient"]
            x = all_X[i].unsqueeze(0)
            y = all_y[i].item()
            
            # Get latent representation
            mu, _ = model.encode(x)
            latent_vectors.append(mu.cpu().numpy().flatten())
            labels.append(y)
            patient_ids.append(patient)
            
            # Calculate attributions
            target = int(all_y[i])
            attr, _ = ig.attribute(
                x, 
                baseline, 
                target=target, 
                return_convergence_delta=True,
                n_steps=100,
                internal_batch_size=1
            )
            
            attr = attr.squeeze(0).cpu().numpy()
            group_name = le.inverse_transform([target])[0]
            attributions_group[group_name].append(attr)
            details_group[group_name].append({
                "cell_id": cell_id,
                "patient": patient,
                "attribution": attr
            })
    
    # Save latent representations
    latent_df = pd.DataFrame(
        np.vstack(latent_vectors), 
        columns=[f"dim_{i}" for i in range(latent_dim)]
    )
    latent_df["group"] = [le.inverse_transform([label])[0] for label in labels]
    latent_df["patient"] = patient_ids
    latent_df.to_csv(os.path.join(cell_type_dir, "latent_representations.csv"), index=False)

    # Calculate attribution summaries
    avg_attr = {}
    pos_percent = {}
    for group, attr_list in attributions_group.items():
        attr_array = np.stack(attr_list, axis=0)
        avg_attr[group] = np.mean(attr_array, axis=0)
        pos_percent[group] = (attr_array > 0).mean(axis=0) * 100

    # Process results in a deterministic order by sorting group names
    for group in sorted(avg_attr.keys()):
        df_summary = pd.DataFrame({
            "gene": gene_names,
            "average_importance": avg_attr[group],
            "positive_percentage": pos_percent[group]
        })

        # Sort consistently
        df_summary = df_summary.reindex(df_summary['average_importance'].abs().sort_values(ascending=False).index)
        output_file = os.path.join(cell_type_dir, f"BetaVAE_{group}.csv")
        df_summary.to_csv(output_file, index=False)

        gene_details = []
        for gi, gene in enumerate(gene_names):
            cells = []
            patients = []
            for detail in details_group[group]:
                if detail["attribution"][gi] > 0:
                    cells.append(detail["cell_id"])
                    patients.append(detail["patient"])
        
            # Sort to ensure consistent output strings
            cells = sorted(list(set(cells)))
            patients = sorted(list(set(patients)))

            gene_details.append({
                "gene": gene,
                "positive_percentage": pos_percent[group][gi],
                "contributing_cells": ";".join(cells),
                "contributing_patients": ";".join(patients)
            })

        df_gene_details = pd.DataFrame(gene_details)
        # Sort consistently
        df_gene_details = df_gene_details.reindex(df_gene_details['positive_percentage'].abs().sort_values(ascending=False).index)
        details_output = os.path.join(cell_type_dir, f"BetaVAE_Gene_info_{group}.csv")
        df_gene_details.to_csv(details_output, index=False)

log.info("Completed successfully!\n")