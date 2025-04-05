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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from captum.attr import IntegratedGradients
from statsmodels.stats.multitest import multipletests

# Set seeds for reproducibility
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
        
        # Apply standardization to each feature
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(df[gene_names].values).astype(np.float32)
        
        self.y = df["Group_enc"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

############################################
class BetaVAEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes=2, dropout_rate=0.1):
        super(BetaVAEClassifier, self).__init__()
        # Reset seed right before weight initialization for reproducible weights
        torch.manual_seed(42)
        
        # Encoder with more layers and batch normalization for stability
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder with more layers and batch normalization
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.bn_dec1 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout_dec1 = nn.Dropout(dropout_rate)
        
        self.fc_dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn_dec2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_dec2 = nn.Dropout(dropout_rate)
        
        self.fc_dec3 = nn.Linear(hidden_dim, input_dim)
        
        # Classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Initialize weights using a deterministic method
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization to ensure deterministic behavior.
        Uses a more conservative initialization to prevent exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use a more conservative initialization
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout2(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clip logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reset seed for sampling - this is crucial for VAE reproducibility
        torch.manual_seed(42)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        # Add epsilon for numerical stability
        return mu + eps * (std + 1e-6)

    def decode(self, z):
        h = F.relu(self.bn_dec1(self.fc_dec1(z)))
        h = self.dropout_dec1(h)
        
        h = F.relu(self.bn_dec2(self.fc_dec2(h)))
        h = self.dropout_dec2(h)
        
        recon = self.fc_dec3(h)
        return recon

    def forward(self, x):
        # Check for NaN inputs and log warning
        if torch.isnan(x).any():
            logging.warning("NaN values detected in input data!")
            
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits

############################################
def loss_function(recon, x, mu, logvar, logits, labels, beta=1.0, kl_weight_schedule=None, epoch=None):
    """
    Beta-VAE loss function with improved numerical stability.
    Includes gradient clipping and KL weight scheduling.
    """
    # Use mean squared error for reconstruction loss
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence with numerical stability improvements
    # Using the exact formula for the KL of two Gaussians
    # log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2)/(2*sigma2^2) - 1/2
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Apply KL annealing if schedule is provided
    if kl_weight_schedule is not None and epoch is not None:
        beta = beta * kl_weight_schedule[epoch]
    
    # Apply beta weight with a more conservative approach at the beginning
    kl_weighted = beta * kl_loss
    
    # Classification loss
    class_loss = F.cross_entropy(logits, labels)
    
    # Total loss with gradient clipping
    total_loss = recon_loss + kl_weighted + class_loss
    
    # Check for NaN values in loss components
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
        logging.warning(f"NaN/Inf detected in loss calculation! recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}, class_loss: {class_loss.item()}")
        
        # Fall back to only using components that aren't NaN
        components = [
            recon_loss if not (torch.isnan(recon_loss).any() or torch.isinf(recon_loss).any()) else torch.tensor(0.0, device=recon_loss.device),
            kl_weighted if not (torch.isnan(kl_weighted).any() or torch.isinf(kl_weighted).any()) else torch.tensor(0.0, device=kl_weighted.device),
            class_loss if not (torch.isnan(class_loss).any() or torch.isinf(class_loss).any()) else torch.tensor(0.0, device=class_loss.device)
        ]
        
        # Recompute without the NaN components
        total_loss = sum(components)
    
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
            
            # Skip batches with NaN values
            if torch.isnan(batch_x).any():
                continue
                
            mu, _ = model.encode(batch_x)
            latent_vectors.append(mu.cpu().numpy())
    
    # Check if we have any valid latent vectors
    if not latent_vectors:
        return {
            'variance_per_dim': np.zeros(model.fc_mu.out_features),
            'coef_variation': np.zeros(model.fc_mu.out_features),
            'variance_ratio': 0.0,
            'active_dimensions': 0
        }
    
    # Concatenate all latent vectors
    latent_vectors = np.vstack(latent_vectors)
    
    # Calculate variance for each dimension
    dim_variances = np.var(latent_vectors, axis=0)
    
    # Calculate coefficient of variation for each dimension
    dim_means = np.abs(np.mean(latent_vectors, axis=0))
    dim_means = np.where(dim_means < 1e-10, 1e-10, dim_means)  # Avoid division by zero
    coef_variation = dim_variances / dim_means
    
    # Metrics to track disentanglement
    variance_ratio = np.max(dim_variances) / np.min(dim_variances) if np.min(dim_variances) > 0 else 0
    active_dims = np.sum(dim_variances > 0.01)
    
    return {
        'variance_per_dim': dim_variances,
        'coef_variation': coef_variation,
        'variance_ratio': variance_ratio,
        'active_dimensions': active_dims
    }

############################################
# Hyperparameters
batch_size = 64  # Reduced batch size for better stability
num_epochs = 50  # Increased epochs to allow for warm-up
learning_rate = 5e-4  # Reduced learning rate for stability
latent_dim = 32
hidden_dim = 256
beta_start = 0.01  # Start with a very small beta value
beta_end = 2.0  # End with a reasonable beta value (less than original 4.0)
dropout_rate = 0.2  # Regularization
weight_decay = 1e-5  # L2 regularization
gradient_clip_val = 1.0  # For gradient clipping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KL annealing schedule for more stable training
def create_kl_annealing_schedule(start=0.0, end=1.0, epochs=num_epochs):
    """Create a schedule for KL weight annealing"""
    return np.linspace(start, end, epochs)

kl_schedule = create_kl_annealing_schedule(0.0, 1.0, num_epochs)

############################################
# File paths
adata_file = '/orcd/home/002/yongheng/PNF/PNF_GEO_complete.h5ad'
base_output_dir = '/orcd/home/002/yongheng/PNF/BetaVAE_Results_20250401_Yongheng_v2'

# Create the directory if it doesn't exist
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir, exist_ok=True)
    log.info(f"Created output directory: {base_output_dir}")
else:
    log.info(f"Output directory already exists: {base_output_dir}")

adata = sc.read(adata_file)
log.info(f"Shape of the dataset: {adata.shape}\n")

# Instead of using scanpy's built-in functions, implement our own normalization
# to have more control over the process
log.info("Performing custom normalization...")

# Sort cell types for consistent processing order
cell_types = sorted(adata.obs["Cell_type"].unique())

for cell_type in cell_types:
    log.info(f"Processing cell type: {cell_type}\n")

    cell_type_dir = os.path.join(base_output_dir, f"{cell_type}")

    os.makedirs(cell_type_dir, exist_ok=True)
    adata_subset = adata[adata.obs["Cell_type"] == cell_type].copy()
    
    # Extract the expression matrix
    expr_array = adata_subset.X.A if hasattr(adata_subset.X, "A") else adata_subset.X
    
    # Check for NaN or Inf values before creating the DataFrame
    if np.isnan(expr_array).any() or np.isinf(expr_array).any():
        log.warning(f"NaN or Inf values detected in the expression matrix for {cell_type}")
        # Replace NaN/Inf with zeros
        expr_array = np.nan_to_num(expr_array, nan=0.0, posinf=0.0, neginf=0.0)
    
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

    # Check for any NaN values in the dataset before creating the dataloader
    if expr_df[gene_names].isna().any().any():
        log.warning(f"NaN values found in {cell_type} dataset. Filling with zeros.")
        expr_df[gene_names] = expr_df[gene_names].fillna(0)
    
    # Create dataset and dataloader
    dataset = CellDataset(expr_df, gene_names)
    
    # Added a deterministic random number generator for DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=g,
        worker_init_fn=seed_worker
    )

    # Initialize the Beta-VAE model with dropout for regularization
    model = BetaVAEClassifier(
        input_dim, 
        hidden_dim, 
        latent_dim, 
        dropout_rate=dropout_rate
    ).to(device)
    
    # Use Adam optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay  # L2 regularization
    )
    
    # Add learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True, 
        min_lr=1e-6
    )

    best_loss = float("inf")
    patience = 8  # Increased patience
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
    
    # Track learning rates
    all_lrs = []

    for epoch in range(num_epochs):
        # Reset seed at the beginning of each epoch with a unique but deterministic value
        epoch_seed = 42 + epoch
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
        random.seed(epoch_seed)
        
        # Calculate current beta value based on annealing schedule
        beta_value = beta_start + (beta_end - beta_start) * kl_schedule[epoch]
        
        total_loss = 0
        recon_losses = 0
        kl_losses = 0
        class_losses = 0
        
        model.train()
        batch_count = 0  # To count valid batches
        
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            # Set a unique but deterministic seed for each batch
            batch_seed = 42 + epoch * 1000 + batch_idx
            torch.manual_seed(batch_seed)
            
            # Check for NaN values in the batch
            if torch.isnan(batch_x).any():
                log.warning(f"NaN values detected in batch {batch_idx} of epoch {epoch}. Skipping...")
                continue
                
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            
            try:
                recon, mu, logvar, logits = model(batch_x)
                loss, r_loss, k_loss, c_loss = loss_function(
                    recon, batch_x, mu, logvar, logits, batch_y, 
                    beta=beta_value, 
                    kl_weight_schedule=kl_schedule,
                    epoch=epoch
                )
                
                # Skip if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    log.warning(f"NaN/Inf loss detected in batch {batch_idx} of epoch {epoch}. Skipping...")
                    continue
                    
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
                recon_losses += r_loss.item() * batch_x.size(0)
                kl_losses += k_loss.item() * batch_x.size(0)
                class_losses += c_loss.item() * batch_x.size(0)
                batch_count += 1
                
            except RuntimeError as e:
                log.error(f"Runtime error in batch {batch_idx} of epoch {epoch}: {str(e)}")
                continue
        
        # Only calculate average if we have processed at least one batch
        if batch_count > 0:
            avg_loss = total_loss / (batch_count * batch_size)
            avg_recon_loss = recon_losses / (batch_count * batch_size)
            avg_kl_loss = kl_losses / (batch_count * batch_size)
            avg_class_loss = class_losses / (batch_count * batch_size)
            
            # Store losses for plotting
            all_losses['total'].append(avg_loss)
            all_losses['recon'].append(avg_recon_loss)
            all_losses['kl'].append(avg_kl_loss)
            all_losses['class'].append(avg_class_loss)
            
            # Store current learning rate
            all_lrs.append(optimizer.param_groups[0]['lr'])
            
            log.info(f"Epoch {epoch+1}/{num_epochs} for cell type {cell_type}, "
                     f"Beta: {beta_value:.4f}, "
                     f"Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
                     f"KL: {avg_kl_loss:.4f}, Class: {avg_class_loss:.4f}, "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Update learning rate scheduler
            scheduler.step(avg_loss)
            
            # Check for improvement - we're looking for >0.5% improvement
            if avg_loss < best_loss * 0.995:
                best_loss = avg_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            log.warning(f"No valid batches in epoch {epoch} for cell type {cell_type}")
            patience_counter += 1
            
            # Add placeholder values to keep the arrays aligned
            all_losses['total'].append(np.nan)
            all_losses['recon'].append(np.nan)
            all_losses['kl'].append(np.nan)
            all_losses['class'].append(np.nan)
            all_lrs.append(optimizer.param_groups[0]['lr'])

        if patience_counter >= patience:
            log.info(f"Early stopping triggered for cell type {cell_type} at epoch {epoch+1}\n")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        log.warning(f"No best model found for {cell_type}. Using the last model state.")

    # Calculate disentanglement metrics
    disent_metrics = calculate_disentanglement(model, dataloader, device)
    
    # Save disentanglement metrics
    pd.DataFrame({
        'dimension': range(latent_dim),
        'variance': disent_metrics['variance_per_dim'],
        'coef_variation': disent_metrics['coef_variation']
    }).to_csv(os.path.join(cell_type_dir, "disentanglement_metrics.csv"), index=False)
    
    # Save learning rates
    pd.DataFrame({'learning_rate': all_lrs}).to_csv(
        os.path.join(cell_type_dir, "learning_rates.csv"), index=False
    )
    
    log.info(f"Disentanglement metrics for {cell_type}:")
    log.info(f"  Variance ratio: {disent_metrics['variance_ratio']:.4f}")
    log.info(f"  Active dimensions: {disent_metrics['active_dimensions']} / {latent_dim}")

    # Save loss history
    pd.DataFrame(all_losses).to_csv(os.path.join(cell_type_dir, "training_losses.csv"), index=False)

    # Reset seed for evaluation
    torch.manual_seed(42)  
    model.eval()
    
    # Skip evaluation if the model didn't train properly
    if np.isnan(best_loss) or np.isinf(best_loss):
        log.warning(f"Skipping evaluation for {cell_type} due to training issues.")
        continue
    
    ig = IntegratedGradients(classifier_forward)
    # Creating a reference point for IG algorithm 
    baseline = torch.zeros((1, input_dim)).to(device)

    attributions_group = {cls: [] for cls in le.classes_}
    details_group = {cls: [] for cls in le.classes_}

    # Process in smaller batches to avoid OOM issues
    batch_size_eval = 32
    
    all_latent_vectors = []
    all_labels = []
    all_patient_ids = []
    
    # Process in batches
    for i in range(0, len(dataset), batch_size_eval):
        batch_end = min(i + batch_size_eval, len(dataset))
        
        # Extract batch data
        batch_X = torch.tensor(dataset.X[i:batch_end], device=device)
        batch_y = torch.tensor(dataset.y[i:batch_end], device=device)
        batch_cell_ids = dataset.df.index[i:batch_end].tolist()
        batch_patients = dataset.df.loc[batch_cell_ids, "Patient"].tolist()
        
        # Skip if NaN values are detected
        if torch.isnan(batch_X).any():
            log.warning(f"NaN values detected in evaluation batch. Skipping batch {i}-{batch_end}...")
            continue
            
        with torch.no_grad():
            # Get latent representations
            mu, _ = model.encode(batch_X)
            
            # Store latent vectors and metadata
            all_latent_vectors.extend(mu.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_patient_ids.extend(batch_patients)
        
        # Calculate attributions for this batch
        for j in range(len(batch_X)):
            # Set a unique but deterministic seed
            sample_seed = 42 + i + j
            torch.manual_seed(sample_seed)
            
            cell_id = batch_cell_ids[j]
            patient = batch_patients[j]
            x = batch_X[j].unsqueeze(0)
            y = batch_y[j].item()
            
            # Skip if NaN values are detected
            if torch.isnan(x).any():
                continue
                
            try:
                # Calculate attributions
                target = int(y)
                attr, _ = ig.attribute(
                    x, 
                    baseline, 
                    target=target, 
                    return_convergence_delta=True,
                    n_steps=50,  # Reduced from 100 for better stability
                    internal_batch_size=1
                )
                
                attr = attr.squeeze(0).cpu().numpy()
                
                # Check for NaN values in attributions
                if np.isnan(attr).any() or np.isinf(attr).any():
                    log.warning(f"NaN/Inf values in attributions for cell {cell_id}. Skipping...")
                    continue
                    
                group_name = le.inverse_transform([target])[0]
                attributions_group[group_name].append(attr)
                details_group[group_name].append({
                    "cell_id": cell_id,
                    "patient": patient,
                    "attribution": attr
                })
            except Exception as e:
                log.error(f"Error calculating attributions for cell {cell_id}: {str(e)}")
                continue
    
    # Save latent representations
    if all_latent_vectors:
        latent_df = pd.DataFrame(
            np.vstack(all_latent_vectors), 
            columns=[f"dim_{i}" for i in range(latent_dim)]
        )
        latent_df["group"] = [le.inverse_transform([label])[0] for label in all_labels]
        latent_df["patient"] = all_patient_ids
        latent_df.to_csv(os.path.join(cell_type_dir, "latent_representations.csv"), index=False)
    else:
        log.warning(f"No valid latent representations for {cell_type}")

    # Calculate attribution summaries
    avg_attr = {}
    pos_percent = {}
    for group, attr_list in attributions_group.items():
        if not attr_list:
            log.warning(f"No attributions for group {group} in cell type {cell_type}")
            continue
            
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