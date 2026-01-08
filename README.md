# Vehicle Sales Prediction 

This project focuses on predicting vehicle sales amounts using machine learning models such as **Ridge Regression**, **Random Forest**, and optionally **LightGBM / XGBoost**, with experiment tracking handled by **MLflow**.

---

## 1. Prerequisites

Make sure the following tools are installed on your system:

### Required Tools
- **Python 3.12+**
- **pip**
- **Git**
- (Optional but recommended) **Anaconda / Miniconda**

Check versions:
```bash
python --version
pip --version

```

## 1. Prerequisites
Create virtual environment. From the project root directory 
```bash
python -m venv venv
venv\Scripts\activate (Windows)
source venv/bin/activate (MacOS/Linux)
```

Then install project dependencies:
```bash
pip install -r requirements.txt
```

## 2. Codes
The full notebook is available in `src/model_prediction.ipynb`, to run in local, restart kernel and use the virtual environment. The html `src/model_prediction.html`

The data should be placed in directory `ROOT/data/`, for e.g. `ROOT/data/DatiumTrain.rpt`

To start jupyter notebook (Note be sure to install jupyter first if needed, i.e. `pip install jupyter`), run this in the terminal
```bash
jupyter notebook
```

## 3. Mlflow
Mlflow is available and setup under the model trainer object, located in `src/model_trainer.py`. When the notebooks is run while the model training is executed, the mlflow will be triggered. 

ALl the params, metrics and model information are being sent to mlflow for now

To view the mlflow ui, run in the terminal:
```bash
mlflow ui
```

Author: Hon Huin Chin <honhuin95@gmail.com>