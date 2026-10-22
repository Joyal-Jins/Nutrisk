# 🥗 ML-Based Packaged Food Nutrition Risk Analyzer

An interactive machine learning web application that analyzes the nutritional composition of packaged food products and classifies their nutritional risk level as **Low, Moderate, or High**. The system uses nutritional information to help users better understand the potential nutritional risks associated with packaged foods.

## 📌 Project Overview

Packaged food products may contain varying amounts of sugar, fat, saturated fat, salt, sodium, protein, fiber, and other nutrients. Understanding these values can help consumers make more informed food choices.

The ML-Based Packaged Food Nutrition Risk Analyzer uses a trained machine learning model to evaluate nutritional data and predict a nutrition risk category. The web application displays the predicted category to make the results easier to understand.

The system also includes a child-focused nutrition warning feature that checks whether selected nutrient values exceed specified thresholds for children based on the user's input age and the configured reference limits.

## ✨ Features

- **Nutrition Risk Prediction:** Classifies packaged food products into Low, Moderate, or High nutritional risk categories.
- **Machine Learning:** Uses a trained machine learning model to predict risk from nutritional features.
- **Nutrient Analysis:** Evaluates available nutritional values such as energy, fat, saturated fat, sugars, salt, sodium, protein, and fiber.
- **Child-Focused Nutrition Warning:** Highlights selected nutrient values that exceed configured child-focused thresholds.
- **Age-Based Assessment:** Uses the entered age to determine which applicable child-focused nutrient limits should be checked.
- **Interactive Web Interface:** Allows users to enter nutritional information and view prediction results.
- **Easy-to-Understand Results:** Presents the predicted nutrition risk category and relevant warnings.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Web Application:** Streamlit
- **Model Storage:** Joblib / Pickle
- **Dataset:** Open Food Facts nutritional data or the dataset used during model training

*Update this list if your implementation uses different libraries or data sources.*

## ⚙️ System Workflow

1. The user enters the packaged food's nutritional information.
2. The application validates and prepares the input data.
3. The trained machine learning model processes the nutritional features.
4. The system predicts the nutrition risk category: Low, Moderate, or High.
5. The application checks applicable child-focused nutrient thresholds when age-based assessment is enabled.
6. The results, predicted risk grade, and relevant warnings are displayed to the user.

## 📂 Project Structure

```text
Nutrisk/
├── app.py                  # Streamlit web application
├── nutrition_model.pkl     # Trained model (stored locally)
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from Git
└── README.md               # Project documentation
```

The actual project structure may contain additional files for preprocessing, datasets, model training, and evaluation.

**Note:** `nutrition_model.pkl` is excluded from the GitHub repository because it exceeds GitHub's regular file size limit. You must obtain or recreate this file before running the application.

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or another Python version compatible with your dependencies
- pip package manager
- Git
- The trained model file, `nutrition_model.pkl`

### 1. Clone the repository

```bash
git clone https://github.com/Joyal-Jins/Nutrisk.git
cd Nutrisk
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains all required packages and compatible versions.

### 4. Add the trained model

Place the trained `nutrition_model.pkl` file in the project root directory, or update the model-loading path in the application if the file is stored elsewhere.

### 5. Run the application

```bash
streamlit run app.py
```

The application will display a local URL in the terminal. Open that URL in your browser to use the system.

## 📊 Machine Learning Approach

The project uses supervised machine learning to predict a nutritional risk category from selected nutrient features.

The general workflow includes:

- Data collection and loading
- Data cleaning and preprocessing
- Feature selection
- Training and testing the machine learning model
- Model evaluation
- Saving the trained model
- Integrating the model with the web application

The final model and evaluation metrics should be documented according to the actual implementation and experimental results.

## 👶 Child-Focused Nutrition Assessment

The application includes an age-based warning mechanism intended to identify selected nutrient values that exceed configured child-focused thresholds.

This feature is designed to improve awareness of nutrient levels in packaged food products. Thresholds should be based on reliable, age-appropriate nutritional guidance and should account for the relevant serving size and units.

The warnings are informational and should not be interpreted as a medical diagnosis or a complete assessment of a child's diet.

## 🎯 Project Objectives

- To analyze nutritional information from packaged food products.
- To apply machine learning for nutrition risk classification.
- To make nutritional risk information easier to understand.
- To identify selected nutrient values that exceed configured child-focused thresholds.
- To support informed food choices through an interactive web application.

## 🔮 Future Enhancements

- Barcode scanning for packaged food products.
- Integration with a food nutrition database or API.
- Food label image recognition using OCR.
- Improved model evaluation and comparison.
- Visualization of nutrient values and risk categories.
- Deployment as an online web application.
- Expanded age-specific nutritional guidance.

## ⚠️ Disclaimer

This project is intended for educational and informational purposes only. The predicted risk category is based on the model, its training data, and the features provided. It does not replace professional nutritional advice, medical guidance, or official food-label interpretation.

The child-focused warnings depend on the thresholds configured in the application and should not be treated as individualized dietary recommendations.

## 👨‍💻 Author

**Joyal Jins**

GitHub: [@Joyal-Jins](https://github.com/Joyal-Jins)

## 📄 License

No license has been specified yet. If you intend to make the project open source, consider adding an appropriate license, such as the MIT License, after confirming that the project code and datasets can be distributed under those terms.
