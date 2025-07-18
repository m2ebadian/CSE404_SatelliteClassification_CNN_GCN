# **Satellite Image Classification with CNN and GCN**

This project explores satellite image classification using both a **Convolutional Neural Network (CNN)** and a **Graph Convolutional Network (GCN)** on the **EuroSAT** dataset. The goal was to compare how traditional CNNs and graph-based models perform on land-cover classification tasks.

The work was done as part of **CSE404: Machine Learning** at Michigan State University, with a focus on building and experimenting with real models rather than just following templates.

---

## **Project Overview**

- **Dataset:** [EuroSAT](https://github.com/phelber/eurosat), consisting of 27,000 RGB satellite images across 10 land-cover classes (forests, crops, urban, water bodies, etc.).
- **Baseline Model:** ResNet-18 (pretrained on ImageNet, fine-tuned for EuroSAT).
- **Graph Model:** GCN built on top of ResNet-18 feature embeddings, with a k-nearest neighbors (kNN) graph constructed using cosine similarity.
- **Key Goal:** Compare accuracy and generalization between CNN and GCN, and analyze where graph-based methods help or fail in satellite image classification.

---

## **Results**

| Model        | Test Accuracy | Precision | Recall | F1 Score |
|--------------|--------------|-----------|--------|----------|
| **ResNet-18 (CNN)** | **96.7%** | 96.7% | 96.7% | 96.7% |
| **GCN (ResNet features + kNN graph)** | **91.1%** | 91.3% | 91.1% | 91.2% |

- **CNN Strengths:** Excellent generalization and high accuracy across most classes.
- **GCN Insights:** Added relational information between images, but performance suffered due to weak inter-image relationships in EuroSAT and limitations of batch-wise kNN graphs.

---

## **My Contributions**

**Mehrshad Bagherebadian, 3rd year Computer Science Major, baghereb@msu.edu**  
- **Hyperparameter Tuning:** Tuned learning rates, weight decay, and training strategies for both models.
- **CV (Computer Vision):** Implemented preprocessing, augmentation, and trained the ResNet-18 baseline model.
- **Graph-Based Learning:** Built and tested the GCN wrapper on top of ResNet-18, including feature extraction and kNN graph construction.

---

## **Project Structure**
📂 CSE404_MLSatelliteImageClassification
- ├── 📄 CSE404_resnet18_modelCNN.ipynb # Baseline ResNet-18 CNN implementation
- ├── 📄 ML404_resnet18_pygcn_model.ipynb # GCN hybrid model built on ResNet-18 features
- ├── 📄 README.md # Project documentation
- ├── 📄 SatelliteClassificationCNNGNNPresentation.pdf # Final presentation slides
- └── 📄 SS25CSE404_SatelliteClassificationProposal.pdf # Original project proposal


---

## **Running the Notebooks**

1. Clone this repository and open the notebooks in Jupyter or VS Code.
2. Make sure the following libraries are installed:
```bash
pip install torch torchvision torch-geometric scikit-learn matplotlib seaborn
```
- Run:
- CSE404_resnet18_modelCNN.ipynb → Baseline CNN training & evaluation.
- ML404_resnet18_pygcn_model.ipynb → Graph-based GCN training & evaluation.
- Tip: Training on GPU (CUDA) is recommended for faster performance.

---

## **Acknowledgements**

This project was completed as part of **CSE404: Machine Learning** at **Michigan State University**.  
Special thanks to:
- **Dr. Kristen Johnson** for guidance throughout the course.
- My teammates: **Gabriel Treutle, Aaryan Naruka, Vish Challa, Ryan Krupp, and Pratham Pradhan** for their contributions in data preprocessing, training, and evaluation.

---

## **References**

- Helber, P., Bischke, B., Dengel, A., & Borth, D. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*.
- Zhu, X. X., et al. *Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources*. IEEE, 2017.
- Veličković, P., et al. *Graph Attention Networks*. ArXiv, 2017.
- Chen, Y., et al. *Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks*. IEEE Transactions on Geoscience and Remote Sensing, 2016.

---

# **Related Files**

[![Proposal](https://img.shields.io/badge/Proposal-PDF-blue)](SS25CSE404_SatelliteClassificationProposal.pdf)  
[![Presentation](https://img.shields.io/badge/Presentation-PDF-orange)](SatelliteClassificationCNNGNNPresentation.pdf)

---

# Feel free to contact me at **baghereb@msu.edu** with any questions!
