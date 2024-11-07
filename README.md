

# Fenty Beauty Data Analysis

This repository contains a sentiment analysis project on Fenty Beauty product reviews. Using natural language processing techniques, this project aims to evaluate customer sentiment and gain insights into product reception by analyzing text reviews.

## Repository Structure

```
fenty_beauty_data_analysis/
├── reviews/                    # Contains the dataset of Fenty Beauty reviews
│   └── fenty_beauty_reviews.csv
├── src/                        # Source files and notebooks for data analysis and modeling
│   ├── best_model.keras        # Pre-trained model file
│   ├── fenty_beauty_my_model.ipynb          # Notebook for training a custom sentiment analysis model
│   ├── fenty_beauty_VADER_DistilBert.ipynb  # Notebook for VADER and DistilBERT-based analysis
├── README.md                   # Project overview and instructions
└── requirements.txt            # Dependencies for the project
```

## Notebooks Overview

### 1. `fenty_beauty_my_model.ipynb`
This notebook trains a custom sentiment analysis model for Fenty Beauty reviews using LSTM-based neural networks. It includes data preprocessing, model architecture design, training, and evaluation steps. It leverages Keras and TensorFlow for building and fine-tuning the model.

### 2. `fenty_beauty_VADER_DistilBert.ipynb`
This notebook uses VADER and DistilBERT for sentiment analysis, providing a comparative analysis of traditional and transformer-based NLP models. It applies the VADER rule-based sentiment analyzer and fine-tunes DistilBERT for classifying review sentiments.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jessakhi/fenty_beauty_data_analysis.git
   cd fenty_beauty_data_analysis
   ```

2. **Install Dependencies**:
   Install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

The project relies on the following libraries:

```plaintext
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.1
matplotlib==3.6.2
seaborn==0.12.2
jupyter==1.0.0
tensorflow==2.11.0       # For training custom sentiment model (LSTM)
transformers==4.24.0     # For using DistilBERT in NLP
torch==1.13.1            # Required by the transformers library
nltk==3.7                # For VADER sentiment analysis
wordcloud==1.8.1         # For generating word clouds of review text
```

Note: Ensure you have a compatible version of TensorFlow and PyTorch installed for your system.

## Usage

1. **Data Preparation**:
   Place the `fenty_beauty_reviews.csv` file in the `reviews/` directory.

2. **Running Notebooks**:
   Launch Jupyter Notebook and open either of the notebooks in the `src/` folder:
   ```bash
   jupyter notebook
   ```
   - Use `fenty_beauty_my_model.ipynb` for custom model training.
   - Use `fenty_beauty_VADER_DistilBert.ipynb` for VADER and DistilBERT-based analysis.

3. **Model Evaluation**:
   After training, the model can be evaluated on the Fenty Beauty dataset in the `reviews/` directory. Visualizations, including bar plots, pie charts, and word clouds, are generated for deeper insights.

## Analysis & Visualization

This project includes various visualization techniques to help understand the sentiment distribution in the reviews:
- **Loss and Accuracy Curves**: Track training and validation loss over epochs.
- **Sentiment Distribution**: Bar and pie charts show the ratio of positive to negative reviews.
- **Word Clouds**: Highlight frequently used words in positive and negative reviews.

## Results

The model achieves reasonable accuracy on the test set, indicating it can reliably distinguish between positive and negative sentiments in Fenty Beauty reviews. However, the model could benefit from further tuning and larger datasets.

## Future Improvements

1. **Experiment with Model Architectures**: Testing with other architectures like GRU or transformer-based models could improve performance.
2. **Hyperparameter Tuning**: Adjust learning rate, batch size, and number of LSTM layers for optimal results.
3. **Enhanced Data Preprocessing**: More rigorous data cleaning, such as removing stop words, lemmatization, and handling negations, could improve accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
