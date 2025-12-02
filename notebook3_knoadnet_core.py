"""
KnoAD-Net Phase 3: Main Model (FIXED VERSION)
==============================================
Fixes overfitting with:
- Increased dropout
- Label smoothing
- Data augmentation
- L2 regularization
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random

import config
import utils

# ============================================================================
# Data Augmentation
# ============================================================================

class AugmentedADDataset(Dataset):
    """Dataset with augmentation to prevent overfitting"""
    def __init__(self, dataframe, feature_cols, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.augment = augment
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load MRI
        img = np.load(row['processed_path']).astype(np.float32)
        img = torch.FloatTensor(img).permute(2, 0, 1)
        
        # Apply augmentation (only during training)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, [2])
            
            # Random rotation (-10 to +10 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                img = transforms.functional.rotate(img, angle)
            
            # Random brightness adjustment
            if random.random() > 0.5:
                factor = random.uniform(0.85, 1.15)
                img = torch.clamp(img * factor, 0, 1)
            
            # Random noise
            if random.random() > 0.7:
                noise = torch.randn_like(img) * 0.02
                img = torch.clamp(img + noise, 0, 1)
        
        # Get features
        features = torch.FloatTensor(row[self.feature_cols].values.astype(np.float32))
        label = int(row['diagnosis_encoded'])
        
        return {
            'image': img,
            'features': features,
            'label': label
        }

# ============================================================================
# Model Architecture (With Higher Regularization)
# ============================================================================

class CrossModalAttention(nn.Module):
    """Cross-Modal Attention with increased dropout"""
    def __init__(self, d_model=768, n_heads=8, dropout=0.3):  # Increased dropout
        super().__init__()
        self.query_proj = nn.Linear(128, d_model)  # Reduced from 256
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.multihead_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, img_features, tab_features):
        query = self.query_proj(tab_features).unsqueeze(1)
        key = self.key_proj(img_features)
        value = self.value_proj(img_features)
        
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.norm1(query + attn_output)
        
        ffn_output = self.ffn(attn_output)
        output = self.norm2(attn_output + ffn_output)
        
        return output.squeeze(1), attn_weights

class KnoADNet(nn.Module):
    """KnoAD-Net with improved regularization"""
    def __init__(self, n_features, n_classes):
        super().__init__()
        
        # ViT encoder (use Tiny for smaller dataset)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.vit_hidden_dim = 192  # Tiny ViT
        
        # Project ViT features to standard dimension
        self.vit_proj = nn.Linear(192, 768)
        
        # Tabular encoder (smaller to prevent overfitting)
        self.tab_embedding = nn.Sequential(
            nn.Linear(n_features, 128),  # Reduced from 256
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)  # Added dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.3, batch_first=True  # Increased dropout
        )
        self.tab_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # Reduced layers
        
        # Cross-attention
        self.cross_attention = CrossModalAttention(
            d_model=768, n_heads=4, dropout=0.3  # Reduced heads, increased dropout
        )
        
        # Classification head (with heavy regularization)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.BatchNorm1d(256),  # Added batch norm
            nn.Linear(256, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.BatchNorm1d(128),  # Added batch norm
            nn.Linear(128, n_classes)
        )
    
    def forward(self, image, features, return_attention=False):
        # Resize if needed
        if image.shape[-1] != 224:
            image = F.interpolate(image, size=224, mode='bilinear')
        
        # Extract image features
        img_features = self.vit.forward_features(image)
        img_patch_features = img_features[:, 1:, :]
        img_patch_features = self.vit_proj(img_patch_features)
        
        # Extract tabular features
        tab_embed = self.tab_embedding(features).unsqueeze(1)
        tab_features = self.tab_transformer(tab_embed).squeeze(1)
        
        # Cross-modal fusion
        fused_features, attn_weights = self.cross_attention(img_patch_features, tab_features)
        
        # Classification
        class_logits = self.classifier(fused_features)
        
        if return_attention:
            return class_logits, attn_weights
        return class_logits

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images, features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{accuracies.avg:.4f}'})
    
    return losses.avg, accuracies.avg

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    losses = utils.AverageMeter()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            images = batch['image'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            losses.update(loss.item(), images.size(0))
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return losses.avg, acc, all_preds, all_labels

# ============================================================================
# Main Training
# ============================================================================

def main():
    utils.print_section("KNOADNET - PHASE 3: MAIN MODEL (FIXED)")
    
    device = utils.get_device()
    utils.set_seed(config.RANDOM_SEED)
    
    # Load data
    print("\n[1/5] Loading data...")
    splits_dir = config.DIRS['data_splits']
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    test_df = pd.read_csv(splits_dir / 'test.csv')
    
    with open(config.DIRS['data_processed'] / 'dataloader_info.json', 'r') as f:
        dataloader_info = json.load(f)
    
    feature_cols = dataloader_info['feature_cols']
    
    # Create datasets WITH AUGMENTATION
    train_dataset = AugmentedADDataset(train_df, feature_cols, augment=True)  # Augment training
    val_dataset = AugmentedADDataset(val_df, feature_cols, augment=False)
    test_dataset = AugmentedADDataset(test_df, feature_cols, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True   # ← ADD THIS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=False   # validation can keep last batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val:   {len(val_dataset)} samples")
    print(f"✓ Test:  {len(test_dataset)} samples")
    
    # Initialize model
    print("\n[2/5] Building model...")
    model = KnoADNet(
        n_features=dataloader_info['n_features'],
        n_classes=dataloader_info['n_classes']
    ).to(device)
    
    utils.print_model_summary(model)
    
    # Training setup with LABEL SMOOTHING
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.1  # Increased from 0.05
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )
    
    # Training loop
    print(f"\n[3/5] Training for {config.NUM_EPOCHS} epochs...")
    print("="*80)
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-"*80)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        print(f"LR:         {current_lr:.6f}")
        
        # Check for overfitting
        if train_acc - val_acc > 0.20:  # 20% gap
            print(f"⚠ WARNING: Overfitting detected! (Train-Val gap: {train_acc - val_acc:.4f})")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, config.DIRS['models'] / 'knoadnet_best_fixed.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*80)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print("="*80)
    
    # Test evaluation
    print("\n[4/5] Final Evaluation on Test Set")
    print("-"*80)
    
    checkpoint = torch.load(config.DIRS['models'] / 'knoadnet_best_fixed.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print("="*80)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*80)
    
    # Compare with baselines
    print("\n[5/5] Comparison with Baselines")
    print("="*80)
    
    try:
        with open(config.DIRS['results'] / 'concat_fusion_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        baseline_acc = baseline_results['test_accuracy']
        improvement = (test_acc - baseline_acc) * 100
        
        print(f"Best Baseline (Concat Fusion): {baseline_acc:.4f}")
        print(f"KnoAD-Net (Fixed):             {test_acc:.4f}")
        print(f"Improvement:                   {improvement:+.2f}%")
        
        if test_acc > baseline_acc:
            print(f"\n✅ SUCCESS! KnoAD-Net beats baseline by {improvement:.2f}%")
        else:
            print(f"\n⚠ KnoAD-Net did not beat baseline (gap: {improvement:.2f}%)")
            print("   Try: More epochs, different augmentation, or larger dataset")
    
    except FileNotFoundError:
        print("Baseline results not found for comparison")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=dataloader_info['class_names'],
        digits=4
    ))
    
    # Save results
    results = {
        'model': 'KnoAD-Net-Fixed',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(best_val_acc),
        'history': history,
        'improvements': {
            'increased_dropout': True,
            'label_smoothing': True,
            'data_augmentation': True,
            'batch_normalization': True,
            'weight_decay_increased': True
        }
    }
    
    with open(config.DIRS['results'] / 'knoadnet_fixed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    utils.print_section("COMPLETE!")
    print(f"""
Results saved to: {config.DIRS['results']}/knoadnet_fixed_results.json
Model saved to:   {config.DIRS['models']}/knoadnet_best_fixed.pth

Next steps:
1. If test accuracy > baseline: Proceed to Phase 5 (Evaluation)
2. If still overfitting: Try collecting more data or increase regularization further
3. Run: python phase5_evaluation.py
""")

if __name__ == "__main__":
    # Add transforms import at top
    
    main()