import os
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.info("Starting to run...\n")

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
class VAEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes=2):
        super(VAEClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec2 = nn.Linear(hidden_dim, input_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        recon = self.fc_dec2(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits

############################################
def loss_function(recon, x, mu, logvar, logits, labels):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    class_loss = F.cross_entropy(logits, labels)
    return recon_loss + kl_loss + class_loss, recon_loss, kl_loss, class_loss

############################################
def classifier_forward(x):
    mu, logvar = model.encode(x)
    logits = model.classifier(mu)
    return logits


############################################
batch_size = 128
num_epochs = 20
learning_rate = 1e-3
latent_dim = 32
hidden_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
adata_file = '/orcd/home/002/yongheng/PNF/PNF_GEO_complete.h5ad'
base_output_dir = '/orcd/home/002/yongheng/PNF/VAE_Results_20250326_Yongheng_v1'

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

cell_types = adata.obs["Cell_type"].unique()

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

    if "Other" in expr_df["Group"].unique():
        logging.error("Unknown group present in the dataset!!\n")
        sys.exit()
    le = LabelEncoder()
    expr_df["Group_enc"] = le.fit_transform(expr_df["Group"])

    gene_names = adata_subset.var_names.tolist()
    input_dim = len(gene_names)

    dataset = CellDataset(expr_df, gene_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAEClassifier(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_model_state = None
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, logits = model(batch_x)
            loss, r_loss, k_loss, c_loss = loss_function(recon, batch_x, mu, logvar, logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(dataset)
        log.info(f"Epoch {epoch+1}/{num_epochs} for cell type {cell_type}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
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

    model.eval()
    ig = IntegratedGradients(classifier_forward)

    attributions_group = {cls: [] for cls in le.classes_}
    details_group = {cls: [] for cls in le.classes_}

    baseline = torch.zeros((1, input_dim)).to(device)

    all_X = torch.tensor(dataset.X, device=device)
    all_y = torch.tensor(dataset.y, device=device)
    for i in range(len(all_X)):
        cell_id = dataset.df.index[i]
        patient = dataset.df.loc[cell_id, "Patient"]
        x = all_X[i].unsqueeze(0)
        target = int(all_y[i])
        attr, _ = ig.attribute(x, baseline, target=target, return_convergence_delta=True)
        attr = attr.squeeze(0).cpu().numpy()
        group_name = le.inverse_transform([target])[0]
        attributions_group[group_name].append(attr)
        details_group[group_name].append({
            "cell_id": cell_id,
            "patient": patient,
            "attribution": attr
        })

    avg_attr = {}
    pos_percent = {}
    for group, attr_list in attributions_group.items():
        attr_array = np.stack(attr_list, axis=0)
        avg_attr[group] = np.mean(attr_array, axis=0)
        pos_percent[group] = (attr_array > 0).mean(axis=0) * 100

    for group in avg_attr:
        df_summary = pd.DataFrame({
            "gene": gene_names,
            "average_importance": avg_attr[group],
            "positive_percentage": pos_percent[group]
        })

        df_summary = df_summary.reindex(df_summary['average_importance'].abs().sort_values(ascending=False).index)
        output_file = os.path.join(cell_type_dir, f"VAE_{group}.csv")
        df_summary.to_csv(output_file, index=False)

        gene_details = []
        for gi, gene in enumerate(gene_names):
            cells = []
            patients = []
            for detail in details_group[group]:
                if detail["attribution"][gi] > 0:
                    cells.append(detail["cell_id"])
                    patients.append(detail["patient"])
            cells = list(set(cells))
            patients = list(set(patients))

            gene_details.append({
                "gene": gene,
                "positive_percentage": pos_percent[group][gi],
                "contributing_cells": ";".join(cells),
                "contributing_patients": ";".join(patients)
            })

        df_gene_details = pd.DataFrame(gene_details)
        df_gene_details = df_gene_details.reindex(df_gene_details['positive_percentage'].abs().sort_values(ascending=False).index)
        details_output = os.path.join(cell_type_dir, f"VAE_Gene_info_{group}.csv")
        df_gene_details.to_csv(details_output, index=False)

log.info("Completed successfully!\n")