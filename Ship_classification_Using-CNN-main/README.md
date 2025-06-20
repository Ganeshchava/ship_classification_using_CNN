# Ship Classification with Balanced Data and CNNs 🚢

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

Welcome to the official repository for the paper *"Enhancement of Ship Classification Accuracy with Balanced Data and Convolutional Neural Networks"*, presented at the **2025 5th International Conference on Intelligent Technologies (CONIT)** and published in IEEE Xplore (Paper ID: 2429). This project demonstrates a robust framework for ship classification using a balanced dataset and multiple CNN architectures, achieving a state-of-the-art accuracy of **93.03%** with VGG16.

Authors: **R. Darwin Hareesh**, **J. Durga Prasad**, **Ch. Ganesh Kumar**, **Arnab De** (Project Guide)  
Institution: Department of ACSE, Vignan's Foundation for Science, Technology and Research, Guntur, India

## 📖 Overview

This project addresses the challenge of ship classification in maritime imagery by tackling class imbalance and leveraging deep learning. We balanced a dataset of 8,932 images (Cargo, Military, Carrier, Cruise, Tankers) to 15,000 images (3,000 per class) using data augmentation. Four CNN models—Custom CNN, VGG16, ResNet50, and MobileNet—were trained and evaluated, with VGG16 achieving the highest accuracy of 93.03%. A Streamlit web app enables real-time classification, making the framework practical for maritime surveillance.

🔗 **Read the Paper**: [IEEE Xplore Link](https://ieeexplore.ieee.org/document/2429) (replace with actual link once published)

## ✨ Features

- **Data Augmentation**: Balanced dataset to 15,000 images using flips, brightness changes, zooms, and rotations.
- **Multi-Model Evaluation**: Trained and benchmarked four CNN architectures:
  - Custom CNN (82.43% accuracy)
  - VGG16 (93.03% accuracy)
  - ResNet50 (90.20% accuracy)
  - MobileNet (71.40% accuracy)
- **Real-Time Classification**: Streamlit web app for interactive ship classification.
- **Comprehensive Analysis**: Includes accuracy, precision, recall, F1-scores, confusion matrices, and training plots.
- **Reproducible Code**: Scripts for data preprocessing, model training, evaluation, and deployment.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/darwinhareesh/ship-classification-cnn.git
   cd ship-classification-cnn
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   ```
   tensorflow==2.10.0
   numpy==1.23.5
   pandas==1.5.0
   matplotlib==3.6.0
   seaborn==0.12.0
   streamlit==1.10.0
   scikit-learn==1.1.2
   pillow==9.2.0
   ```

## 📊 Dataset

We used the ["Game of Deep Learning Ship Datasets"](https://www.kaggle.com/datasets/arpitjain099/ship-dataset) from Kaggle, which originally contains 8,932 images across five classes:
- Cargo: 2,120 train, 908 test
- Military: 1,167 train, 500 test
- Carrier: 916 train, 392 test
- Cruise: 832 train, 356 test
- Tankers: 1,217 train, 524 test

### Data Augmentation
To address class imbalance, we augmented the dataset to 15,000 images (3,000 per class) using:
- Flips (horizontal)
- Brightness changes (0.4–1.6)
- Zooms (0–30%)
- Rotations (0–40°)

**Note**: Due to GitHub size limits, the dataset is not included in this repository. Download it from Kaggle and place it in the `data/` directory:
```
data/
  train/
    cargo/
    military/
    carrier/
    cruise/
    tankers/
  test/
    cargo/
    military/
    carrier/
    cruise/
    tankers/
```

## 🚀 Usage

### 1. Preprocess and Augment Data
Run the data augmentation script to balance the dataset:
```bash
python scripts/data_augmentation.py
```
This script processes images in `data/train/` and saves the augmented dataset to `data/augmented/`.

### 2. Train Models
Train the four CNN models using the following script:
```bash
python scripts/train_models.py
```
- Outputs model weights to `models/`.
- Generates training plots (accuracy and loss) in `plots/`.

### 3. Evaluate Models
Evaluate the trained models on the test set:
```bash
python scripts/evaluate_models.py
```
- Outputs performance metrics (accuracy, precision, recall, F1-score) and confusion matrices to `results/`.

### 4. Run the Streamlit Web App
Launch the Streamlit app for real-time ship classification:
```bash
streamlit run app.py
```
- Open your browser at `http://localhost:8501`.
- Upload a ship image (224x224 pixels, RGB) to get predictions from all four models.

## 📈 Model Performance

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| Custom CNN  | 82.43%   | 82.35%    | 82.43% | 82.31%   |
| VGG16       | **93.03%** | **93.09%** | **93.03%** | **93.05%** |
| ResNet50    | 90.20%   | 90.28%    | 90.20% | 90.14%   |
| MobileNet   | 71.40%   | 71.50%    | 71.40% | 70.98%   |

### Per-Class F1-Scores
| Model       | Cargo  | Military | Carrier | Cruise | Tankers |
|-------------|--------|----------|---------|--------|---------|
| Custom CNN  | 0.8150 | 0.9100   | 0.9200  | 0.8800 | 0.7900  |
| VGG16       | 0.8763 | 0.9718   | 0.9650  | 0.9472 | 0.8734  |
| ResNet50    | 0.8600 | 0.9500   | 0.9400  | 0.9200 | 0.8300  |
| MobileNet   | 0.6370 | 0.8900   | 0.9100  | 0.8700 | 0.5977  |

## 🌐 Streamlit Web App

The Streamlit app allows real-time ship classification:
- **Input**: Upload a ship image (224x224 pixels, RGB).
- **Output**: Confidence scores from all four models (e.g., a cruise ship image yielded Custom CNN: 32.75% Cruise, MobileNet: 99.27% Carrier).

![Streamlit App Screenshot](/streamlit_screenshot.png)  
*(Replace with an actual screenshot of your app.)*

## 📝 Contributing

We welcome contributions to improve this project! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and ensure your code adheres to PEP 8 style guidelines.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration inquiries, reach out to:
- **R. Darwin Hareesh**: [darwinhareesh@gmail.com](mailto:darwinhareesh@gmail.com)
- **Arnab De** (Project Guide): [ade.ece1990@gmail.com](mailto:ade.ece1990@gmail.com)

---

⭐ If you find this project useful, please give it a star on GitHub!  
Happy coding! 🚀
