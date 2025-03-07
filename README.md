# Parkinson's Disease Detection via Speech Analysis

A multilingual deep learning approach combining Variational Mode Decomposition (VMD) with self-supervised learning for robust, non-invasive early detection of Parkinson's disease from speech signals.

## 📖 Overview

This project develops an advanced system for detecting Parkinson's disease (PD) through the analysis of speech patterns, leveraging the fact that speech disorders appear early in PD progression. Our approach is non-invasive, cost-effective, and works across multiple languages, making it accessible and practical for early screening.

### Key Features

- **Multilingual Support**: Works across Spanish and Italian with potential to extend to other languages
- **Robust Feature Extraction**: Combines Variational Mode Decomposition (VMD) with deep learning
- **Self-Supervised Learning**: Utilizes pre-trained Wav2Vec2.0 model fine-tuned on elderly speech datasets
- **Contrastive Learning**: Enhances feature robustness without domain-specific adaptations

## 🔬 Methodology

Our approach follows a carefully designed pipeline:

1. **Data Preprocessing**
   - Resampling to 16 kHz
   - Silence and noise removal
   - Normalization

2. **Feature Extraction**
   - **Variational Mode Decomposition (VMD)**: Decomposes speech signals into Intrinsic Mode Functions (IMFs) to capture non-linear speech patterns
   - **Self-Supervised Learning**: Using Wav2Vec2.0 to extract deep speech embeddings

3. **Training Strategy**
   - **Pre-training**: Self-supervised learning on healthy elderly speech datasets (Spanish and Italian)
   - **Fine-tuning**: Transfer learning on PD speech datasets
   - **Contrastive Learning**: Enhancing feature discrimination between PD and healthy speech patterns


## 📁 Repository Structure

- `dataset.py`: Contains dataset loading and preprocessing pipelines
- `trainer.py`: Modular trainer for model training and evaluation
- `logger.py`: Custom logging utility for tracking experiments
- `pretrainer.py`: Self-supervised pretraining implementation
- `final_pretrainer.py`: Enhanced pretraining with contrastive loss

## 📊 Datasets

We utilized several datasets for this project:

- **PD-Neurovoz** (Spanish)
- **PD-Italian**
- **VoxLingua107** (Multilingual, for pretraining)
- **CommonVoice** (Multilingual, for pretraining)

## 🔄 Current Progress

- ✅ Data Preprocessing
- ✅ VMD Feature Extraction
- ✅ Model Training and Evaluation
- ✅ Wav2Vec2.0 Pretraining
- ⏳ Contrastive Learning Integration
- ⏳ Cross-language fine tuning

## 💡 Future Work

- Expand to more languages
- Incorporate temporal features for longitudinal analysis
- Develop a lightweight inference model for clinical deployment
- Explore interpretability methods for clinical insights

## 📚 References

1. "NeuroVoz: a Castillian Spanish corpus of parkinsonian speech"
2. "Multilingual evaluation of interpretable biomarkers in PD" - Frontiers
3. "Robust and Complex Pathological Speech Analysis" - IEEE
4. "Towards a Generalizable Speech Marker for PD Diagnosis" - arXiv
5. "Self-Supervised Contrastive Learning for Medical Time Series"

## 👥 Team

- Arun Prasad TD (CB.EN.U4AIE22004)
- Ganesh Sundhar S (CB.EN.U4AIE22017)
- Hari Krishnan N (CB.EN.U4AIE22020)
- Shruthikaa V (CB.EN.U4AIE22047)
