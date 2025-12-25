import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, log_loss
from tqdm import tqdm
import argparse
import time
from dlrm_model import DLRM_Net

# Check for deepctr_torch
try:
    from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
    from deepctr_torch.models import xDeepFM
    HAS_DEEPCTR = True
except ImportError:
    HAS_DEEPCTR = False

# --- Data Loading ---
def get_data(data_path, batch_size, device):
    print("Loading data...")
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']
    
    data = pd.read_csv(data_path, sep='\t', header=None, names=target + dense_features + sparse_features)
    print(f"Raw data loaded. Shape: {data.shape}")
    # Preprocessing (Unified)
    # Fill NaNs
    print("Filling NaNs...")
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)
    # Label Encoding for Sparse
    print("Encoding and Scaling features...")
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat].astype(str))
    
    # MinMax Scaling for Dense
    print("Scaling dense features...")
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # Feature Columns for xDeepFM
    fixlen_feature_columns = []
    if HAS_DEEPCTR:
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=10, dtype='float32')
                                  for feat in sparse_features] + \
                                 [DenseFeat(feat, 1) for feat in dense_features]

    # Prepare Tensor Data
    # We will create a single tensor with [Sparse, Dense] order to match xDeepFM's expectation if we construct it that way.
    # xDeepFM notebook: sparse then dense in fixlen_feature_columns.
    
    # Concatenate features
    # Sparse features are integers, Dense are floats.
    # We'll convert everything to float for the single tensor, but keep track of indices.
    # DLRM needs Long for sparse. We will cast back inside the loop.
    
    X_sparse = data[sparse_features].values # (N, 26)
    X_dense = data[dense_features].values   # (N, 13)
    y = data[target].values                 # (N, 1)
    
    # Concatenate: Sparse first, then Dense
    X_full = np.hstack([X_sparse, X_dense])
    
    # Split: 80-20 train-val
    X_train, X_val, y_train, y_val = train_test_split(X_full, y, test_size=0.2, random_state=2025, shuffle=False)
    
    # Create Datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # DLRM
    # Calculate vocab size for each sparse feature
    ln_emb = []
    for feat in sparse_features:
        ln_emb.append(data[feat].nunique())
    
    return train_loader, val_loader, fixlen_feature_columns, ln_emb

# --- Training Functions ---

def train_dlrm_model(args, train_loader, val_loader, ln_emb, device):
    print(f"Initializing DLRM with ln_emb={ln_emb}")
    
    # DLRM_Net configuration
    m_spa = 16
    ln_bot = [13, 512, 256, 64, m_spa] # Input 13, Output m_spa
    
    # Calculate interaction output dimension
    # n_sparse = 26
    # n_dense_transformed = 1
    # total_vectors = 27
    # interaction_dim = 27 * 28 // 2 = 378
    # concatenated with dense vector (16) -> 394
    ln_top = [394, 512, 256, 1]
    
    model = DLRM_Net(
        m_spa=m_spa,
        ln_emb=ln_emb,
        ln_bot=ln_bot,
        ln_top=ln_top,
        arch_interaction_op="dot",
        arch_interaction_itself=True,
        sigmoid_top=len(ln_top) - 2,
        loss_function="bce"
    )
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DLRM model initialized with {total_params} trainable parameters.")
    
    # Calculate Embedding parameters
    embedding_params = 0
    for p in model.emb_l.parameters():
        embedding_params += p.numel()
    print(f"DLRM Embedding parameters: {embedding_params}")
    print(f"DLRM MLP parameters: {total_params - embedding_params}")
    print(f"Embedding Ratio: {embedding_params/total_params:.2%}")
    
    criterion = torch.nn.BCELoss()
    
    # Split parameters for optimizers
    # EmbeddingBag with sparse=True requires SparseAdam or Adagrad
    embeddings = [p for n, p in model.named_parameters() if "emb_l" in n]
    rest = [p for n, p in model.named_parameters() if "emb_l" not in n]
    
    optimizer_dense = torch.optim.Adam(rest, lr=0.001)
    # Use SparseAdam for embeddings
    optimizer_sparse = torch.optim.SparseAdam(embeddings, lr=0.001)
    
    writer = SummaryWriter(log_dir=f'logs/dlrm_{int(time.time())}')
    
    print("Starting DLRM training...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_preds = []
        train_targets = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                
                # Split x into sparse and dense
                # x is [Sparse (26), Dense (13)]
                x_sparse = x[:, :26].long()
                x_dense = x[:, 26:]
                
                # Prepare sparse inputs for DLRM_Net
                # lS_i: list of tensors (Batch,)
                # lS_o: list of tensors (Batch,) - offsets
                lS_i = [x_sparse[:, i] for i in range(x_sparse.shape[1])]
                lS_o = [torch.arange(x_sparse.shape[0], device=device) for _ in range(x_sparse.shape[1])]
                
                optimizer_dense.zero_grad()
                optimizer_sparse.zero_grad()
                
                output = model(x_dense, lS_o, lS_i)
                # output is (Batch, 1)
                
                loss = criterion(output, y)
                loss.backward()
                
                optimizer_dense.step()
                optimizer_sparse.step()
                
                loss_val = loss.item()
                total_loss += loss_val
                writer.add_scalar('Loss/train_batch', loss_val, global_step)
                global_step += 1
                
                # Collect for epoch metrics
                probs = output.detach().cpu().numpy()
                train_preds.extend(probs)
                train_targets.extend(y.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            
        avg_train_loss = total_loss / len(train_loader)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_acc = accuracy_score(train_targets, (train_preds > 0.5).astype(int))
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Metric/train_auc', train_auc, epoch)
        writer.add_scalar('Metric/train_acc', train_acc, epoch)
        
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        predictions = []
        targets = []
        val_loss = 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x = x.to(device)
                y = y.to(device)
                x_sparse = x[:, :26].long()
                x_dense = x[:, 26:]
                
                lS_i = [x_sparse[:, i] for i in range(x_sparse.shape[1])]
                lS_o = [torch.arange(x_sparse.shape[0], device=device) for _ in range(x_sparse.shape[1])]
                
                output = model(x_dense, lS_o, lS_i)
                loss = criterion(output, y)
                val_loss += loss.item()
                
                predictions.extend(output.cpu().numpy())
                targets.extend(y.cpu().numpy())
        
        end_time = time.time()
        inference_time = end_time - start_time
        max_memory = 0
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
        avg_val_loss = val_loss / len(val_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        val_auc = roc_auc_score(targets, predictions)
        val_ll = log_loss(targets, predictions)
        val_acc = accuracy_score(targets, (predictions > 0.5).astype(int))
        
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Metric/val_auc', val_auc, epoch)
        writer.add_scalar('Metric/val_acc', val_acc, epoch)
        writer.add_scalar('Performance/val_inference_time', inference_time, epoch)
        writer.add_scalar('Performance/val_peak_memory', max_memory, epoch)
        
        print(f"Val Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, Time: {inference_time:.2f}s, Mem: {max_memory:.2f}MB")

    # Final Classification Report
    print("\nFinal Classification Report (Validation Set):")
    y_pred_binary = (predictions > 0.5).astype(int)
    print(classification_report(targets, y_pred_binary, digits=4))

    # Save model
    torch.save(model.state_dict(), f'dlrm_model_epoch_{args.epochs}.pth')
    print("Model saved.")
    writer.close()

def train_xdeepfm_model(args, train_loader, val_loader, feature_columns, device):
    if not HAS_DEEPCTR:
        print("Error: deepctr_torch is not installed.")
        return

    print("Initializing xDeepFM...")
    dnn_feature_columns = feature_columns
    linear_feature_columns = feature_columns
    
    model = xDeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=[400, 400, 400, 400],
        cin_layer_size=[200, 200, 50],
        cin_split_half=True,
        cin_activation='relu',
        l2_reg_linear=0.0001,
        l2_reg_embedding=0.0001,
        l2_reg_dnn=0.0001,
        l2_reg_cin=0.0001,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0.0,
        dnn_activation='relu',
        dnn_use_bn=False,
        task='binary',
        device=device
    )
    sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"xDeepFM model initialized with {sum_params} trainable parameters.")
    # 计算嵌入表参数
    embedding_params = 0
    linear_embedding_params = 0

    print("\n--- Embedding Parameters Details ---")
    for name, param in model.named_parameters():
        if 'embedding' in name:
            # print(f"{name}: {param.shape}") # 可选：打印每个嵌入表的形状
            if 'linear_model' in name:
                linear_embedding_params += param.numel()
            else:
                embedding_params += param.numel()

    print(f"Deep Side Embedding Params: {embedding_params}")
    print(f"Linear Side Embedding Params: {linear_embedding_params}")
    print(f"Total Embedding Params: {embedding_params + linear_embedding_params}")
    print(f"Embedding Ratio: {(embedding_params + linear_embedding_params)/sum_params:.2%}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.BCELoss()
    
    writer = SummaryWriter(log_dir=f'logs/xdeepfm_{int(time.time())}')
    
    print("Starting xDeepFM training...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        train_preds = []
        train_targets = []
        train_loss_sum = 0.0
        train_BCEloss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]") as pbar:
            for x, y in pbar:
                x = x.to(device).float()
                y = y.to(device).float()
                
                optimizer.zero_grad()
                y_pred = model(x).squeeze()
                loss = loss_func(y_pred, y.squeeze())
                
                reg_loss = model.get_regularization_loss()

                total_loss = loss + reg_loss
                total_loss.backward()
                # (loss + reg_loss).backward()
                optimizer.step()
                
                loss_val = loss.item()
                train_BCEloss += loss_val
                train_loss_sum += total_loss.item()
                writer.add_scalar('Loss/train_batch_logloss', loss_val, global_step)
                writer.add_scalar('Loss/train_batch_total', total_loss.item(), global_step)
                global_step += 1
                
                # Collect for epoch metrics
                train_preds.extend(y_pred.detach().cpu().numpy())
                train_targets.extend(y.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_logloss = train_BCEloss / len(train_loader)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_acc = accuracy_score(train_targets, (train_preds > 0.5).astype(int))
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_epoch_logloss', avg_train_logloss, epoch)
        writer.add_scalar('Metric/train_auc', train_auc, epoch)
        writer.add_scalar('Metric/train_acc', train_acc, epoch)
        
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        predictions = []
        targets = []
        val_loss = 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                x = x.to(device).float()
                y = y.to(device).float()
                
                y_pred = model(x).squeeze()
                loss = loss_func(y_pred, y.squeeze())
                val_loss += loss.item()
                
                predictions.extend(y_pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
        
        end_time = time.time()
        inference_time = end_time - start_time
        max_memory = 0
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
        avg_val_loss = val_loss / len(val_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        val_auc = roc_auc_score(targets, predictions)
        val_ll = log_loss(targets, predictions)
        val_acc = accuracy_score(targets, (predictions > 0.5).astype(int))
        
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Metric/val_auc', val_auc, epoch)
        writer.add_scalar('Metric/val_acc', val_acc, epoch)
        writer.add_scalar('Performance/val_inference_time', inference_time, epoch)
        writer.add_scalar('Performance/val_peak_memory', max_memory, epoch)
        
        print(f"Val Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, Time: {inference_time:.2f}s, Mem: {max_memory:.2f}MB")

    # Final Classification Report
    print("\nFinal Classification Report (Validation Set):")
    y_pred_binary = (predictions > 0.5).astype(int)
    print(classification_report(targets, y_pred_binary, digits=4))

    # Save model
    torch.save(model.state_dict(), f'xdeepfm_model_epoch_{args.epochs}.pth')
    print("Model saved.")
    writer.close()

def main():
    parser = argparse.ArgumentParser(description="Train DLRM or xDeepFM")
    parser.add_argument('--model', type=str, required=True, choices=['dlrm', 'xdeepfm', 'both'], help='Model to train')
    parser.add_argument('--data_path', type=str, default='train.txt', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    # if args.model == 'dlrm':
    #     args.batch_size = 16
    train_loader, val_loader, feature_columns, max_sparse_val = get_data(args.data_path, args.batch_size, args.device)
    
    if args.model == 'dlrm':
        train_dlrm_model(args, train_loader, val_loader, max_sparse_val, args.device)
    elif args.model == 'xdeepfm':
        train_xdeepfm_model(args, train_loader, val_loader, feature_columns, args.device)
    elif args.model == 'both':
        train_dlrm_model(args, train_loader, val_loader, max_sparse_val, args.device)
        train_xdeepfm_model(args, train_loader, val_loader, feature_columns, args.device)

if __name__ == '__main__':
    main()
