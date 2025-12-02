"""
KnoAD-Net Phase 2: Baseline Models
Trains three baseline models for comparison
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config
import utils

# ============================================================================
# Dataset Class
# ============================================================================

class ADDataset(Dataset):
    """Alzheimer's Disease Dataset"""
    def __init__(self, dataframe, feature_cols):
        self.df = dataframe.reset_index(drop=True)
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.load(row['processed_path']).astype(np.float32)
        img = torch.FloatTensor(img).permute(2, 0, 1)
        features = torch.FloatTensor(row[self.feature_cols].values.astype(np.float32))
        label = int(row['diagnosis_encoded'])
        
        return {'image': img, 'features': features, 'label': label}

# ============================================================================
# Model Architectures
# ============================================================================

class ViTBaseline(nn.Module):
    """Vision Transformer for MRI classification"""
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
        
        self.vit.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(x, size=224, mode='bilinear')
        return self.vit(x)

class TabTransformer(nn.Module):
    """Tabular Transformer for cognitive scores"""
    def __init__(self, n_features, n_classes, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        
        self.feature_embedding = nn.Linear(n_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.feature_embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)

class ConcatFusion(nn.Module):
    """Simple concatenation fusion baseline"""
    def __init__(self, n_features, n_classes):
        super().__init__()
        
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        
        self.tab_encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(384 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, image, features):
        if image.shape[-1] != 224:
            image = torch.nn.functional.interpolate(image, size=224, mode='bilinear')
        
        img_features = self.vit(image)
        tab_features = self.tab_encoder(features)
        fused = torch.cat([img_features, tab_features], dim=1)
        
        return self.classifier(fused)

# ============================================================================
# Training Functions
# ============================================================================

def train_vit(model, train_loader, val_loader, device, num_epochs=20):
    """Train ViT baseline"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean()
            
            train_loss.update(loss.item(), images.size(0))
            train_acc.update(acc.item(), images.size(0))
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}', 'acc': f'{train_acc.avg:.4f}'})
        
        # Validate
        model.eval()
        val_loss = utils.AverageMeter()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                val_loss.update(loss.item(), images.size(0))
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc.avg)
        history['val_loss'].append(val_loss.avg)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg:.4f}")
        print(f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.DIRS['models'] / 'vit_baseline_best.pth')
            print(f"✓ Saved best model (acc: {val_acc:.4f})")
    
    return history, best_val_acc

def train_tabular(model, train_loader, val_loader, device, num_epochs=30):
    """Train TabTransformer baseline"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean()
            
            train_loss.update(loss.item(), features.size(0))
            train_acc.update(acc.item(), features.size(0))
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}', 'acc': f'{train_acc.avg:.4f}'})
        
        # Validate
        model.eval()
        val_loss = utils.AverageMeter()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                val_loss.update(loss.item(), features.size(0))
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc.avg)
        history['val_loss'].append(val_loss.avg)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg:.4f}")
        print(f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.DIRS['models'] / 'tab_baseline_best.pth')
            print(f"✓ Saved best model (acc: {val_acc:.4f})")
    
    return history, best_val_acc

def train_fusion(model, train_loader, val_loader, device, num_epochs=30):
    """Train concatenation fusion baseline"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            images = batch['image'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean()
            
            train_loss.update(loss.item(), images.size(0))
            train_acc.update(acc.item(), images.size(0))
            pbar.set_postfix({'loss': f'{train_loss.avg:.4f}', 'acc': f'{train_acc.avg:.4f}'})
        
        # Validate
        model.eval()
        val_loss = utils.AverageMeter()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, features)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                
                val_loss.update(loss.item(), images.size(0))
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc.avg)
        history['val_loss'].append(val_loss.avg)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg:.4f}")
        print(f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.DIRS['models'] / 'concat_fusion_best.pth')
            print(f"✓ Saved best model (acc: {val_acc:.4f})")
    
    return history, best_val_acc

def evaluate_model(model, test_loader, device, model_type='vit'):
    """Evaluate model on test set"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = utils.AverageMeter()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            labels = batch['label'].to(device)
            
            if model_type == 'vit':
                images = batch['image'].to(device)
                outputs = model(images)
            elif model_type == 'tab':
                features = batch['features'].to(device)
                outputs = model(features)
            else:  # fusion
                images = batch['image'].to(device)
                features = batch['features'].to(device)
                outputs = model(images, features)
            
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            
            test_loss.update(loss.item(), labels.size(0))
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    return test_loss.avg, test_acc, all_preds, all_labels

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    utils.print_section("KNOADNET - PHASE 2: BASELINE MODELS")
    
    # Setup
    utils.set_seed(config.RANDOM_SEED)
    device = utils.get_device()
    
    # Load dataloader info
    with open(config.DIRS['data_processed'] / 'dataloader_info.json', 'r') as f:
        dataloader_info = json.load(f)
    
    # Load data
    print("\n[1/4] Loading Data...")
    train_df = pd.read_csv(config.DIRS['data_splits'] / 'train.csv')
    val_df = pd.read_csv(config.DIRS['data_splits'] / 'val.csv')
    test_df = pd.read_csv(config.DIRS['data_splits'] / 'test.csv')
    
    feature_cols = dataloader_info['feature_cols']
    
    train_dataset = ADDataset(train_df, feature_cols)
    val_dataset = ADDataset(val_df, feature_cols)
    test_dataset = ADDataset(test_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"✓ Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    # Baseline 1: ViT-Only
    print("\n[2/4] Training ViT Baseline...")
    vit_model = ViTBaseline(num_classes=config.N_CLASSES).to(device)
    vit_history, vit_best_val = train_vit(vit_model, train_loader, val_loader, device, num_epochs=20)
    
    vit_model.load_state_dict(torch.load(config.DIRS['models'] / 'vit_baseline_best.pth'))
    vit_test_loss, vit_test_acc, vit_preds, vit_labels = evaluate_model(vit_model, test_loader, device, 'vit')
    
    vit_results = {
        'model': 'ViT-Baseline',
        'test_accuracy': float(vit_test_acc),
        'test_loss': float(vit_test_loss),
        'best_val_accuracy': float(vit_best_val),
        'history': vit_history
    }
    
    with open(config.DIRS['results'] / 'vit_baseline_results.json', 'w') as f:
        json.dump(vit_results, f, indent=2)
    
    print(f"\n✓ ViT Baseline - Test Accuracy: {vit_test_acc:.4f}")
    
    # Baseline 2: TabTransformer
    print("\n[3/4] Training TabTransformer Baseline...")
    tab_model = TabTransformer(n_features=dataloader_info['n_features'], n_classes=config.N_CLASSES).to(device)
    tab_history, tab_best_val = train_tabular(tab_model, train_loader, val_loader, device, num_epochs=30)
    
    tab_model.load_state_dict(torch.load(config.DIRS['models'] / 'tab_baseline_best.pth'))
    tab_test_loss, tab_test_acc, tab_preds, tab_labels = evaluate_model(tab_model, test_loader, device, 'tab')
    
    tab_results = {
        'model': 'TabTransformer-Baseline',
        'test_accuracy': float(tab_test_acc),
        'test_loss': float(tab_test_loss),
        'best_val_accuracy': float(tab_best_val),
        'history': tab_history
    }
    
    with open(config.DIRS['results'] / 'tab_baseline_results.json', 'w') as f:
        json.dump(tab_results, f, indent=2)
    
    print(f"\n✓ TabTransformer Baseline - Test Accuracy: {tab_test_acc:.4f}")
    
    # Baseline 3: Concatenation Fusion
    print("\n[4/4] Training Concatenation Fusion Baseline...")
    fusion_model = ConcatFusion(n_features=dataloader_info['n_features'], n_classes=config.N_CLASSES).to(device)
    fusion_history, fusion_best_val = train_fusion(fusion_model, train_loader, val_loader, device, num_epochs=30)
    
    fusion_model.load_state_dict(torch.load(config.DIRS['models'] / 'concat_fusion_best.pth'))
    fusion_test_loss, fusion_test_acc, fusion_preds, fusion_labels = evaluate_model(fusion_model, test_loader, device, 'fusion')
    
    fusion_results = {
        'model': 'Concat-Fusion-Baseline',
        'test_accuracy': float(fusion_test_acc),
        'test_loss': float(fusion_test_loss),
        'best_val_accuracy': float(fusion_best_val),
        'history': fusion_history
    }
    
    with open(config.DIRS['results'] / 'concat_fusion_results.json', 'w') as f:
        json.dump(fusion_results, f, indent=2)
    
    print(f"\n✓ Concatenation Fusion - Test Accuracy: {fusion_test_acc:.4f}")
    
    # Comparison
    comparison_df = pd.DataFrame([
        {'Model': 'ViT-Only', 'Test Accuracy': vit_test_acc, 'Val Accuracy': vit_best_val},
        {'Model': 'TabTransformer', 'Test Accuracy': tab_test_acc, 'Val Accuracy': tab_best_val},
        {'Model': 'Concat Fusion', 'Test Accuracy': fusion_test_acc, 'Val Accuracy': fusion_best_val}
    ])
    
    comparison_df.to_csv(config.DIRS['results'] / 'baseline_comparison.csv', index=False)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Val Accuracy'], width, label='Validation', alpha=0.8)
    ax.bar(x + width/2, comparison_df['Test Accuracy'], width, label='Test', alpha=0.8)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.DIRS['visualizations'] / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
    
    utils.print_section("BASELINE TRAINING COMPLETE!")
    print(f"""
Results Summary:
----------------
✓ ViT-Only:           {vit_test_acc:.4f}
✓ TabTransformer:     {tab_test_acc:.4f}
✓ Concat Fusion:      {fusion_test_acc:.4f}

Best Baseline: {comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']}
Best Accuracy: {comparison_df['Test Accuracy'].max():.4f}

Next Step: Run phase3_knoadnet.py
Goal: Beat {comparison_df['Test Accuracy'].max():.4f} with cross-attention fusion!
""")

if __name__ == "__main__":
    main()
