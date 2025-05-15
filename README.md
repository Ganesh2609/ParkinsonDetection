# Parkinson's Disease Detection via Speech Analysis

A cutting-edge multilingual deep learning approach combining Variational Mode Decomposition (VMD) with self-supervised learning for robust, non-invasive early detection of Parkinson's disease from speech signals.

## 📖 Overview

This project develops an advanced system for detecting Parkinson's disease (PD) through the analysis of speech patterns, leveraging the fact that speech disorders appear early in PD progression. Our approach is non-invasive, cost-effective, and works across multiple languages, making it accessible and practical for early screening.

### Key Features

- **Multilingual Support**: Works across Spanish and Italian with potential to extend to other languages
- **Advanced Feature Extraction**: Combines Variational Mode Decomposition (VMD) with deep learning
- **Self-Supervised Learning**: Utilizes pre-trained Wav2Vec2.0 model fine-tuned on elderly speech datasets
- **Contrastive Learning**: Enhances feature robustness without domain-specific adaptations
- **Multi-Model Approach**: Implements Co-attention, ConvNeXt, ViT, and LSTM architectures

## 🔬 Methodology

### Data Preprocessing Pipeline
1. Resampling to 16 kHz
2. Silence and noise removal  
3. Normalization
4. LMDB database creation for efficient data loading

### Feature Extraction
- **Variational Mode Decomposition (VMD)**: Decomposes speech signals into Intrinsic Mode Functions (IMFs)
- **Spectrogram Analysis**: STFT and Mel-spectrogram representations
- **Self-Supervised Learning**: Using Wav2Vec2.0 to extract deep speech embeddings

### Training Strategy
1. **Pre-training**: Self-supervised learning on healthy elderly speech datasets
2. **Fine-tuning**: Transfer learning on PD speech datasets
3. **Contrastive Learning**: Product quantization with contrastive loss

## 🏗️ Architecture

### Model Variants Implemented

1. **Co-attention Model** (`model.py`/`model2.py`)
   - Multi-modal fusion of audio and image features
   - Transformer-based co-attention mechanism
   - Dual encoder architecture

2. **ConvNeXt** 
   - Modified for multi-channel spectrogram input
   - Pre-trained weights adaptation

3. **Vision Transformer (ViT)**
   - Google's ViT adapted for spectrogram analysis
   - Custom channel modifications

4. **LSTM Classifier**
   - Bidirectional LSTM with projection
   - Separate image and audio encoders
   - Temporal feature extraction

5. **Wav2Vec2.0 Pretraining**
   - Self-supervised learning on elderly speech
   - Product quantization
   - Contrastive loss implementation

## 📁 Repository Structure

```

├── Direct Model Training/
│   ├── Coattention model/
│   │   ├── model.py           # Basic co-attention model
│   │   ├── model2.py          # Co-attention with Wav2Vec2
│   │   └── training_model.py  # Training scripts
│   ├── Convx Net/
│   │   └── training_model.py  # ConvNeXt training
│   ├── LSTM/
│   │   ├── model.py           # LSTM classifier
│   │   └── architecture_testing.ipynb
│   ├── ViT Google/
│   │   └── training_model.py  # Vision Transformer
│   └── Wave2vec/
│       ├── dataset.py         # Dataset classes
│       └── runner_audio.py    # Audio training runner
├── Trainer/
│   ├── trainer.py            # Modular trainer class
│   ├── logger.py             # Logging utilities
│   └── dataset*.py           # Various dataset implementations
├── Final Fine Tuning/
│   ├── trainer.py           # Enhanced trainer
│   └── logger.py            # Training logger
├── Pretraining wave2vec 2.0/
│   ├── dataset.py           # Pretraining dataset
│   └── final_pretrainer.py  # Contrastive pretraining
└── Train data/
    └── Graphs/              # Training visualizations
    └── Logs/              # Training logs
```

## 📊 Results & Visualizations

### Training Performance

<div align="center">

![Co-attention Reconstructed STFT](Train%20data/Graphs/coattention_reconstructed_stft.png)
*Co-attention Model - Reconstructed STFT Features*

![Co-attention Reconstructed STFT Wave2Vec](Train%20data/Graphs/coattention_reconstructed_stft_wave2vec.png)
*Co-attention Model - Reconstructed STFT with Wave2Vec Features*

### Model Comparisons

![ConvNeXt Mel Only](Train%20data/Graphs/convxnet_mel_only.png)
*ConvNeXt - Mel Spectrogram Only*

![ConvNeXt STFT Only](Train%20data/Graphs/convxnet_stft_only.png)
*ConvNeXt - STFT Only*

![ConvNeXt Mel STFT Combined](Train%20data/Graphs/convxnet_mel_stft_both.png)
*ConvNeXt - Combined Mel & STFT Features*

### Vision Transformer Performance

![ViT Google Mel No Pretraining](Train%20data/Graphs/vit_google_mel_no_pretraining.png)
*Vision Transformer - Mel Spectrogram (No Pretraining)*

![ViT Google Mel Only](Train%20data/Graphs/vit_google_mel_only.png)
*Vision Transformer - Mel Spectrogram Only*

![ViT Google STFT No Pretraining](Train%20data/Graphs/vit_google_stft_no_pretraining.png)
*Vision Transformer - STFT (No Pretraining)*

![ViT Google STFT Only](Train%20data/Graphs/vit_google_stft_only.png)
*Vision Transformer - STFT Only*

![ViT Google Mel STFT Combined](Train%20data/Graphs/vit_google_mel_stft_both.png)
*Vision Transformer - Combined Features*

![ViT Google Mel STFT No Pretraining](Train%20data/Graphs/vit_google_mel_stft_no_pretraining.png)
*Vision Transformer - Combined Features (No Pretraining)*

</div>

## 💻 Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ParkinsonDetection.git
cd ParkinsonDetection

# Install dependencies
pip install -r requirements.txt
```

### Training Example

```python
# Train Co-attention Model
from Trainer.trainer import ModularTrainer
from Coattention model.model2 import CoattentionModel
from Trainer.dataset4 import get_data_loaders

# Initialize model
model = CoattentionModel()

# Load data
train_loader, test_loader = get_data_loaders(
    lmdb_path="path/to/vmd.lmdb",
    batch_size=5,
    num_workers=12
)

# Setup training
trainer = ModularTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=24,
    device='cuda'
)

# Train
trainer.train()
```

### Pretraining Wav2Vec2.0

```python
from Pretraining.final_pretrainer import main

# Run pretraining with contrastive loss
main()
```

## 🔧 Technical Details

### Key Libraries
- PyTorch
- Transformers (HuggingFace)
- TorchAudio
- LMDB
- NumPy, Pandas, Matplotlib

### Model Parameters
- **Co-attention**: ~142M parameters
- **ConvNeXt**: ~49M parameters  
- **ViT**: ~88M parameters
- **LSTM**: ~13M parameters
- **Wav2Vec2.0**: ~315M parameters

### Hyperparameters
- Learning Rate: 1e-5 to 2e-4
- Batch Size: 5-32
- Epochs: 16-50
- Optimizer: Adam/AdamW
- Scheduler: ReduceLROnPlateau

## 📊 Datasets

- **PD-Neurovoz** (Spanish PD speech)
- **PD-Italian** (Italian PD speech)
- **VoxLingua107** (Multilingual for pretraining)
- **CommonVoice** (Multilingual for pretraining)

## 🔄 Current Progress

- ✅ Data Preprocessing Pipeline
- ✅ VMD Feature Extraction
- ✅ Multi-Model Training Framework
- ✅ Wav2Vec2.0 Self-Supervised Pretraining
- ✅ Contrastive Learning Integration
- ⏳ Cross-language Fine Tuning
- ⏳ Clinical Deployment Optimization

## 💡 Future Work

- Expand to additional languages (German, French, Mandarin)
- Implement attention visualization for interpretability
- Develop lightweight models for edge deployment
- Create web interface for clinical testing
- Explore multi-task learning for severity assessment

## 📚 Key References

1. "NeuroVoz: a Castillian Spanish corpus of parkinsonian speech" - Pattern Recognition Letters, 2023
2. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" - NeurIPS 2020
3. "Product Quantization for Nearest Neighbor Search" - IEEE TPAMI
4. "Co-attention Networks for Multimodal Learning" - CVPR 2022
5. "Vision Transformer" - ICLR 2021

## 🏆 Acknowledgments

This project was developed as part of the Biomedical Signal Processing course at Amrita Vishwa Vidyapeetham. Special thanks to our faculty advisors and the dataset contributors.

## 👥 Team

- **Ganesh Sundhar S** (CB.EN.U4AIE22017)
- **Arun Prasad TD** (CB.EN.U4AIE22004)
- **Hari Krishnan N** (CB.EN.U4AIE22020)
- **Shruthikaa V** (CB.EN.U4AIE22047)

---
*Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham*
