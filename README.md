# CompTIA DataAI Training Projects

## 📚 CompTIA DataAI Training Portfolio

This repository contains the projects, practical exercises, and machine learning experiments completed during my **CompTIA DataAI Training Program**.

The training provided hands-on experience with **Data Science, Machine Learning, Deep Learning, Natural Language Processing, Time-Series Forecasting, Computer Vision, and MLOps**.

The projects in this repository demonstrate my learning journey from fundamental data preprocessing and sequence modeling to deep learning, NLP, and model deployment concepts.

---

## 👨‍💻 About Me

**Sonu Kumar**
M.Sc. Data Science
Dayananda Sagar University

**GitHub:** [github.com/sonukr-62](https://github.com/sonukr-62)
**LinkedIn:** [linkedin.com/in/sonukumarroy62](https://www.linkedin.com/in/sonukumarroy62)

---

# 🎯 Training Objectives

The main objectives of these projects were to gain practical experience in:

* Data preprocessing
* Exploratory data analysis
* Machine Learning fundamentals
* Deep Learning
* Convolutional Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Long Short-Term Memory (LSTM)
* Natural Language Processing (NLP)
* Computer Vision
* Time-Series Forecasting
* Data Augmentation
* Model Evaluation
* MLflow
* Model Registry
* FastAPI
* Model Deployment
* Model Retraining

---

# 📂 Repository Contents

## 1. 🧠 CNN – Brain Tumor Classification

**Project:** `CNN_Brain_Tumor_Project.ipynb`

This project implements a **Convolutional Neural Network (CNN)** for brain tumor image classification.

The project uses a brain tumor dataset containing four classes:

* Glioma
* Meningioma
* No Tumor
* Pituitary

### Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* ImageDataGenerator
* CNN

### Workflow

```text
Dataset
   ↓
Image Preprocessing
   ↓
Image Rescaling
   ↓
Training / Validation Split
   ↓
CNN Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Model Saving & Loading
```

### CNN Architecture

```text
Input Image
224 × 224 × 3
       ↓
Conv2D (32 filters)
       ↓
MaxPooling
       ↓
Conv2D (64 filters)
       ↓
MaxPooling
       ↓
Conv2D (128 filters)
       ↓
MaxPooling
       ↓
Flatten
       ↓
Dense (128)
       ↓
Softmax (4 Classes)
```

The model was trained using:

* Adam Optimizer
* Categorical Cross-Entropy
* Accuracy
* 10 Training Epochs

The training notebook achieved approximately **99.60% best training accuracy** and **95.64% best validation accuracy**.

---

# 2. 🖼️ Data Augmentation

**Project:** `Data_Agumentation.ipynb`

This project demonstrates different techniques for increasing the diversity of image training data using image augmentation.

### Techniques Covered

* Image Rotation
* Horizontal Flipping
* Brightness Adjustment
* Contrast Adjustment
* Sharpness Adjustment
* Gaussian Blur
* Zoom / Crop
* Gaussian Noise

### Workflow

```text
Original Image
      ↓
Data Augmentation
      ↓
 ┌────┼────┬────┬────┐
 ↓    ↓    ↓    ↓    ↓
Rotate Flip Brightness Contrast Blur
      ↓
Augmented Images
```

The project also visualizes the original and augmented images to understand how augmentation changes the training data.

---

# 3. 🔢 RNN – Next Number Prediction

**Project:** `RNN_predict_next_number_project_1.ipynb`

This project demonstrates how a **Simple Recurrent Neural Network (RNN)** can learn a numerical sequence and predict the next number.

The sequence used is:

```text
1, 2, 3, 4, 5, 6, ... , 100
```

A sliding-window approach is used.

For example:

```text
Input → [1, 2, 3]
Output → 4

Input → [2, 3, 4]
Output → 5

Input → [3, 4, 5]
Output → 6
```

### Model

```text
Input Sequence
      ↓
SimpleRNN (50 units)
      ↓
Dense (1)
      ↓
Next Number
```

### Configuration

* Window Size: 3
* SimpleRNN Units: 50
* Optimizer: Adam
* Loss: Mean Squared Error
* Epochs: 500

This project demonstrates the basic concept of sequence learning using RNNs.

---

# 4. 🔤 RNN – A to Z Next Character Prediction

**Project:** `RNN_AtoZ.ipynb`

This project applies an RNN to an alphabet sequence.

The alphabet is converted into numerical values:

```text
A → 0
B → 1
C → 2
...
Z → 25
```

A three-character window is used to predict the next character.

Examples:

```text
ABC → D
BCD → E
CDE → F
DEF → G
```

### Model

```text
Input
  ↓
SimpleRNN (50 units)
  ↓
Dense (26)
  ↓
Softmax
  ↓
Next Character
```

### Configuration

* Sequence Window: 3
* RNN Units: 50
* Output Classes: 26
* Optimizer: Adam
* Loss: Sparse Categorical Cross-Entropy
* Epochs: 500

This project demonstrates how RNNs can learn patterns from sequential categorical data.

---

# 5. 📝 Next Word Prediction Using Simple RNN

**Project:** `Next_Word_Prediction_using_Simple_RNN_project_2_and_3.ipynb`

This project demonstrates **Natural Language Processing and sequence prediction** using a Simple RNN.

The text is first converted into numerical tokens using a tokenizer.

### NLP Pipeline

```text
Text Data
   ↓
Tokenization
   ↓
Word Indexing
   ↓
N-Gram Sequence Generation
   ↓
Padding
   ↓
Input / Target Separation
   ↓
One-Hot Encoding
   ↓
Embedding
   ↓
SimpleRNN
   ↓
Softmax
   ↓
Next Word Prediction
```

### Model Components

* Tokenizer
* N-Gram sequences
* Padding
* Embedding Layer
* SimpleRNN
* Dense Softmax Layer

### Configuration

* Embedding Dimension: 10
* RNN Units: 64
* Optimizer: Adam
* Loss: Categorical Cross-Entropy
* Epochs: 300

The project demonstrates how a neural network can learn relationships between words and generate a predicted next word based on previously entered text.

---

# 6. 🌦️ LSTM – Jena Climate Time-Series Forecasting

**Project:** `LSTM_Jena_Climate_Dataset.ipynb`

This project focuses on **time-series forecasting using LSTM networks**.

The Jena Climate dataset contains multivariate weather measurements collected at regular time intervals.

The project works with variables including:

* Temperature
* Pressure
* Relative Humidity
* Wind Velocity

### Data Processing

```text
Raw Climate Data
       ↓
Date-Time Processing
       ↓
Resampling
       ↓
Feature Selection
       ↓
Normalization
       ↓
Train / Validation / Test Split
       ↓
Sliding Window
       ↓
LSTM Models
       ↓
Forecasting
       ↓
Evaluation
```

### Models / Experiments

The notebook explores multiple sequence-learning approaches including:

* Basic LSTM
* Stacked LSTM
* Bidirectional LSTM
* Multivariate LSTM
* Many-to-One LSTM
* Many-to-Many LSTM
* Autoencoder-based anomaly detection
* Dropout comparison

This project provides practical exposure to using LSTM networks for sequential and time-dependent data.

---

# 7. 🔤 NLP Text Preprocessing

**Project:** `NLP_DetaAI.ipynb`

This project demonstrates fundamental **Natural Language Processing preprocessing techniques**.

### Techniques Covered

#### Text Normalization

* Lowercasing
* Removing unnecessary characters
* Punctuation handling
* Whitespace cleaning

#### Tokenization

* Word Tokenization
* Character Tokenization

#### Stop Word Removal

Common words that may not provide significant information are removed from the text.

#### Stemming

The project demonstrates **Porter Stemming**.

Example:

```text
playing
played
plays
```

can be reduced toward a common stem.

#### Lemmatization

The project also demonstrates **WordNet Lemmatization**, which attempts to transform words into their meaningful base forms.

### NLP Pipeline

```text
Raw Text
   ↓
Normalization
   ↓
Tokenization
   ↓
Stop Word Removal
   ↓
Stemming
   ↓
Lemmatization
   ↓
Processed Text
```

---

# 8. ⚙️ MLOps – Spam Message Detection

**Project:** `MLOps.ipynb`

This project demonstrates the transition from a machine learning model to a more complete **MLOps workflow**.

The project uses a spam message classification model based on:

* TF-IDF
* Logistic Regression

### Machine Learning Pipeline

```text
Text Dataset
     ↓
Preprocessing
     ↓
TF-IDF
     ↓
Logistic Regression
     ↓
Model Evaluation
```

### MLOps Workflow

```text
Model Development
       ↓
Experiment Tracking
       ↓
MLflow
       ↓
Model Logging
       ↓
Model Registry
       ↓
Model Loading
       ↓
FastAPI
       ↓
API Prediction
       ↓
Deployment
       ↓
New Data Evaluation
       ↓
Retraining
       ↓
New Model Version
```

### Technologies Used

* Python
* Scikit-learn
* TF-IDF
* Logistic Regression
* MLflow
* Joblib
* FastAPI
* Uvicorn
* Ngrok

The project demonstrates concepts such as:

* Experiment tracking
* Model versioning
* Model registry
* API-based prediction
* Model deployment
* Model evaluation on new data
* Model retraining

---

# 🧩 Overall Learning Journey

The projects in this repository represent a progression from fundamental concepts to more advanced AI workflows.

```text
                    DATA & AI TRAINING
                           │
          ┌────────────────┼────────────────┐
          │                │                │
         NLP              CNN              RNN
          │                │                │
          ↓                ↓                ↓
   Text Processing   Image Classification  Sequences
          │                │                │
          ↓                ↓                ↓
   Tokenization      Brain Tumor CNN     Next Number
   Stemming                               Next Character
   Lemmatization                           Next Word
                                             │
                                             ↓
                                           LSTM
                                             │
                                             ↓
                                    Climate Forecasting
                                             │
                                             ↓
                                           MLOps
                                             │
                                             ↓
                                  MLflow + FastAPI
                                             │
                                             ↓
                                      Model Deployment
```

---

# 🛠️ Technologies & Tools

| Category             | Technologies          |
| -------------------- | --------------------- |
| Programming          | Python                |
| Data Processing      | NumPy, Pandas         |
| Visualization        | Matplotlib            |
| Machine Learning     | Scikit-learn          |
| Deep Learning        | TensorFlow, Keras     |
| Computer Vision      | CNN, PIL              |
| Sequence Modeling    | RNN, LSTM             |
| NLP                  | NLTK, Tokenization    |
| Experiment Tracking  | MLflow                |
| Model Management     | MLflow Model Registry |
| API Development      | FastAPI               |
| Server               | Uvicorn               |
| Deployment/Tunneling | Ngrok                 |
| Dataset Sources      | Kaggle                |

---

# 📁 Repository Structure

```text
CompTIA-Data-AI-Training/
│
├── CNN_Brain_Tumor_Project.ipynb
│
├── Data_Agumentation.ipynb
│
├── LSTM_Jena_Climate_Dataset.ipynb
│
├── MLOps.ipynb
│
├── NLP_DetaAI.ipynb
│
├── Next_Word_Prediction_using_Simple_RNN_project_2_and_3.ipynb
│
├── RNN_AtoZ.ipynb
│
├── RNN_predict_next_number_project_1.ipynb
│
├── multiclass classification.ipynb
│
└── README.md
```

> The exact filenames may differ depending on the final organization of the repository.

---

# 🚀 How to Run the Projects

## 1. Clone the Repository

```bash
git clone https://github.com/sonukr-62/<repository-name>.git
```

## 2. Navigate to the Repository

```bash
cd <repository-name>
```

## 3. Install Required Libraries

For the deep learning projects:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn pillow
```

For NLP projects:

```bash
pip install nltk
```

For MLOps:

```bash
pip install mlflow fastapi uvicorn joblib scikit-learn
```

Some notebooks may require additional packages depending on the dataset or environment.

## 4. Open the Notebooks

The notebooks can be opened using:

* Jupyter Notebook
* JupyterLab
* Google Colab
* VS Code

Example:

```bash
jupyter notebook
```

---

# 📊 Skills Demonstrated

Through these projects, I gained practical exposure to:

### Data Science

* Data preprocessing
* Data transformation
* Data normalization
* Dataset splitting
* Data visualization

### Machine Learning

* Classification
* Feature transformation
* Model training
* Model evaluation

### Deep Learning

* Neural networks
* CNN
* RNN
* LSTM
* Embedding layers
* Softmax classification

### Computer Vision

* Image preprocessing
* Image augmentation
* Multi-class image classification
* Brain tumor image classification

### NLP

* Tokenization
* Stop-word removal
* Stemming
* Lemmatization
* N-gram generation
* Next-word prediction

### Time-Series

* Sliding-window sequences
* Multivariate forecasting
* LSTM forecasting
* Anomaly detection

### MLOps

* Experiment tracking
* MLflow
* Model Registry
* Model versioning
* FastAPI
* Model serving
* Retraining workflow

---

# 📈 Learning Progression

The projects helped me understand how different AI techniques are applied to different types of data:

| Data Type          | Project                    | Technique       |
| ------------------ | -------------------------- | --------------- |
| Numerical Sequence | Next Number Prediction     | RNN             |
| Character Sequence | A-Z Prediction             | RNN             |
| Text               | Next Word Prediction       | RNN + NLP       |
| Text               | NLP Preprocessing          | NLTK            |
| Images             | Brain Tumor Classification | CNN             |
| Images             | Data Augmentation          | Computer Vision |
| Time-Series        | Climate Forecasting        | LSTM            |
| Text               | Spam Detection             | ML + MLOps      |

---

# 🎓 Training Outcome

This repository represents my practical learning and implementation during the **CompTIA DataAI Training Program**.

Rather than focusing only on theoretical concepts, the training allowed me to implement complete experiments involving:

**Data → Preprocessing → Model Building → Training → Evaluation → Prediction → Deployment**

The projects helped strengthen my understanding of how different machine learning and deep learning techniques can be selected according to the type of data and problem being solved.

---

# 🔮 Future Improvements

Possible future improvements to these projects include:

* Hyperparameter tuning
* Transfer learning for image classification
* Improving CNN architectures
* Using pretrained models such as VGG, ResNet, or EfficientNet
* Comparing RNN and LSTM architectures
* Using GRU networks
* Transformer-based NLP models
* Improved time-series forecasting
* Dockerizing ML applications
* CI/CD for machine learning
* Cloud deployment
* Automated model monitoring
* Automated model retraining pipelines

---

# 📜 Disclaimer

These projects were developed as part of a **Data & AI training program for educational and learning purposes**.

The notebooks demonstrate experimentation and practical implementation of machine learning and deep learning concepts. They are not intended to be used as medical diagnostic systems or production-critical decision-making systems.

---

# ⭐ Acknowledgement

I would like to thank the instructors and mentors involved in the **CompTIA DataAI Training Program** for providing the learning resources, practical guidance, and hands-on exercises that helped me develop these projects.

---

## 👨‍💻 Author

**Sonu Kumar**

M.Sc Data Science
Dayananda Sagar University

[GitHub](https://github.com/sonukr-62) • [LinkedIn](https://www.linkedin.com/in/sonukumarroy62)

---

⭐ If you find this repository useful, feel free to explore the notebooks and projects.
