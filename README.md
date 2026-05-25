# Leaf Scan — Plant Disease Recognition Expert System

An expert system that diagnoses plant diseases from leaf images by combining a CNN deep learning model with a JSON knowledge base to provide symptoms and treatment recommendations.

## Disclaimer
The model is trained on the PlantVillage dataset which consists of controlled lab images. It may perform differently on real-world field photographs with varying lighting or backgrounds.

## How It Works
Provide a leaf image path to the CLI. The model predicts the disease and the knowledge base returns matching symptoms and recommended treatment.

## Core Components
- **CNN Model** — trained on the PlantVillage dataset to classify 38 plant disease classes
- **Knowledge Base** — JSON file containing symptoms and treatments for each disease
- **CLI App** — takes an image path, runs prediction, and displays diagnosis with treatment advice
- **Training Notebook** — full training pipeline available on [Kaggle](https://www.kaggle.com/code/naveedamarguriro/plant-disease-detector-ai-ccp)

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Kagglehub

## Dataset
The model is trained on the PlantVillage dataset. Run the following to download it:
```bash
python dataset.py
```

Make sure your Kaggle API credentials are configured before running. See [Kaggle API setup](https://www.kaggle.com/docs/api).

## Model
The pre-trained model is not included in this repo due to file size (134MB). To get the model:

- **Option A** — Run the training notebook on Kaggle: [plant-disease-detector-ai-ccp](https://www.kaggle.com/code/naveedamarguriro/plant-disease-detector-ai-ccp)
- **Option B** — Download the dataset and retrain locally using the notebook

Place the generated `plant_disease_model.keras` file in the root project directory.

## Setup

Install dependencies:
```bash
pip install tensorflow numpy kagglehub
```

Download the dataset:
```bash
python dataset.py
```

Run the chatbot:
```bash
python chatbot.py
```

## Usage
1. Run `python chatbot.py`
2. Enter the file path of a leaf image when prompted
3. The system will display the diagnosed disease, confidence score, symptoms and recommended treatment
4. Type `exit` to quit

A few sample images are available in the `test_images/` folder to try out immediately.