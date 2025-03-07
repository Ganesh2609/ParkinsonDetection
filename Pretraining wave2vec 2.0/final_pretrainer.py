import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from dataset import get_data_loaders
from tqdm import tqdm
import json



def main():

    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load Wav2Vec2 model
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
    # model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)


    # # Masking function: randomly mask 10% of the sequence
    # def apply_masking(features, mask_ratio=0.1):
    #     batch_size, seq_len, feature_dim = features.shape
    #     mask = torch.ones_like(features, dtype=torch.bool)
    #     for i in range(batch_size):
    #         num_mask = int(seq_len * mask_ratio)  # Compute number of masked positions
    #         mask_indices = torch.randperm(seq_len)[:num_mask]  # Select random positions
    #         mask[i, mask_indices] = False  # Apply masking
    #     features[~mask] = 0  # Zero out masked positions
    #     return features



    class ProductQuantization(nn.Module):

        def __init__(self, num_groups=16, num_entries=256, feature_dim=1024):
            super().__init__()
            self.num_groups = num_groups
            self.num_entries = num_entries
            self.feature_dim = feature_dim
            # Codebooks: [num_groups, num_entries, feature_dim / num_groups]
            self.codebooks = nn.Parameter(torch.randn(num_groups, num_entries, feature_dim // num_groups))
            #self.codebooks = nn.Parameter(torch.randn(num_groups, num_entries, feature_dim // num_groups).type(torch.float16))

        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            # Reshape input into groups
            x = x.view(batch_size, seq_len, self.num_groups, -1)  # Shape: [batch, seq_len, num_groups, 64]
            # Compute L2 distances between inputs and codebook entries
            distances = (x.unsqueeze(3) - self.codebooks.unsqueeze(0).unsqueeze(0)).pow(2).sum(dim=-1)  # [batch, seq_len, num_groups, num_entries]
            # Get the indices of the closest codebook entry
            indices = distances.argmin(dim=-1)  # Shape: [batch, seq_len, num_groups]
            # **Alternative way to get quantized vectors using indexing instead of gather**
            quantized = self.codebooks[torch.arange(self.num_groups).view(1, 1, -1), indices]  # Shape: [batch, seq_len, num_groups, 64]
            return quantized.view(batch_size, seq_len, -1), indices






    class ContrastiveLoss(nn.Module):

        def __init__(self, temperature=0.1):
            super().__init__()
            self.temperature = temperature

        def forward(self, quantized, true_features):
            quantized = F.normalize(quantized, p=2, dim=-1)
            true_features = F.normalize(true_features, p=2, dim=-1)
            logits = torch.matmul(quantized, true_features.transpose(-1, -2)) / self.temperature
            labels = torch.arange(logits.shape[1], device=logits.device).repeat(logits.shape[0], 1)  # Shape: [batch_size, seq_len]
            loss = F.cross_entropy(logits, labels)
            return loss
    



    # Define Model Components/
    root1 = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Spanish healthy pretrain 1"
    root2 = "E:/Amrita/Subjects/Sem 5/BMSP paper work/Dataset/Italian health pretrain 1"
    dataloader = get_data_loaders(root1=root1, root2=root2, batch_size=8)
    quantizer = ProductQuantization().to(device)
    contrastive_loss = ContrastiveLoss().to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(quantizer.parameters()), lr=1e-4)
    save_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Code base/Newer methodology/Train Data/Checkpoints/pretrain_1.pth' 
    json_path = 'E:/Amrita/Subjects/Sem 5/BMSP paper work/Code base/Newer methodology/Train Data/Logs/pretrain_1.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)



    # Training loop
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in progress_bar:
            batch = batch['Input'].to(device)  # (batch_size, seq_len)
            
            # Step 1: Extract Features (CNN Output)
            extract_features = model.feature_extractor(batch)  # (batch_size, feature_dim, seq_len)
            extract_features = extract_features.transpose(1, 2)  # Ensure correct shape
            
            # Step 2: Apply Feature Projection
            hidden_states, projected_features = model.feature_projection(extract_features)
            
            # Step 3: Apply Masking Properly
            attention_mask = torch.ones(batch.shape, dtype=torch.long, device=device)  # Create full attention mask
            attention_mask = model._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask, add_adapter=False
            )
            masked_features = model._mask_hidden_states(hidden_states, attention_mask=attention_mask)
            
            # Step 4: Pass through Transformer
            encoder_outputs = model.encoder(masked_features, attention_mask=attention_mask)
            transformer_out = encoder_outputs[0]  # Extract last hidden state
            
            # Step 5: Product Quantization
            quantized_output, _ = quantizer(transformer_out)
            
            # Step 6: Compute Contrastive Loss
            loss = contrastive_loss(quantized_output, transformer_out)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            if i % 100 == 0:
                torch.save(model.state_dict(), save_path)
                data['Loss'].append(epoch_loss/(i+1))
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    main()
