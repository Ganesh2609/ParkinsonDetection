import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional
from logger import TrainingLogger

class Quantizer(nn.Module):
    
    def __init__(self, num_codebooks=2, num_codes=320, feature_dim=512):
        super().__init__()
        self.codebooks = nn.Embedding(num_codebooks * num_codes, feature_dim)

    def forward(self, features):
        indices = torch.randint(0, self.codebooks.num_embeddings, (features.shape[0], features.shape[1]), device=features.device)
        quantized_features = self.codebooks(indices)
        return quantized_features, indices

class ModularTrainer:

    def __init__(self, model, train_loader, optimizer=None, scheduler=None, scaler=None,
                 log_path='./logs/training.log', checkpoint_path='./checkpoints',
                 num_epochs=10, mask_prob=0.15, device=None):
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        self.scheduler = scheduler
        self.scaler = scaler
        self.num_epochs = num_epochs
        self.mask_prob = mask_prob
        self.checkpoint_path = checkpoint_path
        
        self.history = {'Training Loss': []}

    def apply_masking(self, features):
        mask = torch.rand(features.shape[:2], device=features.device) < self.mask_prob
        masked_features = features.clone()
        masked_features[mask] = 0
        return masked_features, mask
    
    def contrastive_loss(self, pred_features, quantized_features, tau=0.1):
        pred_features = F.normalize(pred_features, dim=-1)
        quantized_features = F.normalize(quantized_features, dim=-1)
        similarity_matrix = torch.bmm(pred_features, quantized_features.transpose(1, 2)) / tau
        loss = -torch.mean(torch.log(torch.diagonal(F.softmax(similarity_matrix, dim=-1) + 1e-10)))
        return loss
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        quantizer = Quantizer().to(self.device)
        
        with tqdm(self.train_loader, desc=f"Epoch [{self.num_epochs}] (Training)") as t:
            for i, batch in t:
                self.optimizer.zero_grad()
                features = batch.to(self.device)
                masked_features, _ = self.apply_masking(features)
                pred_features = self.model(masked_features)
                quantized_features, _ = quantizer(features)
                loss = self.contrastive_loss(pred_features, quantized_features)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                t.set_postfix({'Loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        self.history['Training Loss'].append(avg_loss)
        self.logger.info(f"Epoch loss: {avg_loss}")
    
    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"Epoch {epoch} starting...")
            self.train_epoch()
            self.save_checkpoint(epoch)
            if self.scheduler:
                self.scheduler.step()
    
    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_path, f'model_epoch_{epoch}.pth')
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, path)
        self.logger.info(f"Checkpoint saved: {path}")
