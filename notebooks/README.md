# 📚 BEATs Trainer Notebooks

Welcome to the BEATs Trainer example notebooks! Choose the right notebook based on your experience level.

## 🚀 Quick Start Notebooks (Recommended for New Users)

### [Quick_Start_Feature_Extraction.ipynb](Quick_Start_Feature_Extraction.ipynb)
**⏱️ 5-10 minutes** | **👤 Beginner-friendly**

Learn how to extract audio features in just a few steps:
- Install and import the library
- Extract features from your audio files  
- Compare audio similarity
- Ready-to-run examples

**Perfect for**: First-time users, quick prototyping, understanding the basics

---

### [Quick_Start_Training.ipynb](Quick_Start_Training.ipynb) 
**⏱️ 15-20 minutes** | **👤 Beginner-friendly**

Train your own custom BEATs model:
- Organize your audio data
- Configure training parameters
- Fine-tune BEATs on your dataset
- Use your trained model

**Perfect for**: Custom audio classification, domain-specific applications

---

## 🧪 Advanced Notebooks (For Experienced Users)

### [Advanced_Feature_Extraction_Analysis.ipynb](Advanced_Feature_Extraction_Analysis.ipynb)
**⏱️ 45-60 minutes** | **👤 Advanced**

Comprehensive feature extraction analysis:
- Detailed comparison of pretrained vs fine-tuned models
- Dimensionality reduction and visualization (PCA, t-SNE, UMAP)
- Quantitative evaluation with silhouette scores
- Advanced plotting and analysis techniques

**Perfect for**: Research, detailed model comparison, publication-quality analysis

---

### [Advanced_ESC50_Fine_Tuning.ipynb](Advanced_ESC50_Fine_Tuning.ipynb)
**⏱️ 60+ minutes** | **👤 Advanced**

Complete ESC-50 fine-tuning walkthrough:
- Comprehensive data preparation and validation
- Advanced training configuration
- Detailed evaluation and metrics
- Model comparison and performance analysis

**Perfect for**: Research reproducibility, advanced training techniques, benchmarking

---

## 🎯 Which Notebook Should I Choose?

### 🔰 **New to BEATs or Audio ML?**
→ Start with **Quick_Start_Feature_Extraction.ipynb**

### 🎯 **Want to train on your own data?** 
→ Use **Quick_Start_Training.ipynb**

### 🧠 **Need detailed analysis or research?**
→ Try the **Advanced_** notebooks

### ⚡ **Just want to see it work quickly?**
→ **Quick_Start_Feature_Extraction.ipynb** (5 minutes)

---

## 📋 Requirements

### For Quick Start Notebooks:
```bash
pip install git+https://github.com/ninanor/beats-trainer.git
```

### For Advanced Notebooks:
```bash
pip install git+https://github.com/ninanor/beats-trainer.git
pip install umap-learn seaborn scikit-learn
```

---

## 💡 Tips for Success

- **Start small**: Use the Quick Start notebooks first
- **Use your own data**: Replace example paths with your audio files
- **GPU recommended**: Training works on CPU but GPU is much faster
- **Check requirements**: Each notebook lists what you need

---

## 🆘 Need Help?

- **Documentation**: [GitHub Repository](https://github.com/ninanor/beats-trainer)
- **Issues**: [Report Problems](https://github.com/ninanor/beats-trainer/issues)
- **Examples**: All notebooks include working examples

Happy audio modeling! 🎵✨
