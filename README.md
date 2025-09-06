# ML Algorithms From Scratch

🚀 This project implements **classic machine learning algorithms from scratch (NumPy only)** and compares them against `scikit-learn` implementations on benchmark datasets.  

The goal is to deeply understand how ML models work under the hood — not just use libraries.

---

## 📂 Project Structure
ML-Algorithms-From-Scratch/

│── Algorithms/ # All scratch implementations

│ ├── logistic_regression.py

│ ├── decision_tree.py

│ ├── knn.py

│ ├── naive_bayes.py

│ ├── perceptron.py

│

│── ML_algorithms_experiments.ipynb # Notebook with Iris + Wine experiments

│── README.md

│── requirements.txt

│── LICENSE


---

## ✅ Implemented Algorithms
- Logistic Regression (binary + OvR multi-class)
- Decision Tree (ID3-like, entropy-based)
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian NB)
- Perceptron (binary classification)

---

## 🧪 Experiments
We tested all algorithms on:

1. **Iris Dataset** 🌸  
   - Multi-class classification (3 classes, 4 features)  
   - Easy dataset, most models achieve ~95–100%  

2. **Wine Dataset** 🍷  
   - More complex, 3-class classification  
   - We used 2 real features (`alcohol`, `color_intensity`) for visualization  
   - Clearer comparison between algorithms  

📊 In the notebook, each algorithm:
- Is trained from scratch  
- Visualizes its decision boundary  
- Compares accuracy with `scikit-learn`  

At the end, we include a **final comparison table and plot** across all algorithms.

---

## 🔧 Installation
Clone this repo:

git clone https://github.com/your-username/ML-Algorithms-From-Scratch.git

cd ML-Algorithms-From-Scratch


Install dependencies:

pip install -r requirements.txt

---

📊 Usage

Open the notebook:

jupyter notebook experiments/all_algorithms.ipynb


Inside, you’ll find:

Training and testing each algorithm.

Visualizations of decision boundaries.

Accuracy comparison between scratch vs sklearn implementations.

---

🎯 Goals

Learn how ML algorithms work under the hood.

Compare performance with sklearn.

Build a foundation for more advanced algorithms in the future (SVM, ensembles, neural nets).

---

📈 Results

On Iris dataset, most algorithms reach high accuracy (>90%).

On Wine dataset, results vary but still competitive with sklearn.

Perceptron works well for binary subsets.

Final comparison is summarized in tables + bar plots in the notebook.

---

🤝 Contributing
Pull requests and suggestions are welcome! If you find an issue, feel free to open one.
