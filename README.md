# Classificador de Sentiments a Xarses Socials en Català (CSXSC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Danie1Arias/sentiment--analysis--catalan--reviews-blue)](https://huggingface.co/datasets/Danie1Arias/sentiment-analysis-catalan-reviews)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Danie1Arias/csxsc-blue)](https://huggingface.co/Danie1Arias/csxsc)

---

## Abstract

This repository contains the complete code and resources for the Master's Thesis project focused on the development and evaluation of a sentiment classification model for Catalan social media content. The primary contribution is the **CSXSC (Classificador de Sentiments a Xarxes Socials en Català)** model, a fine-tuned RoBERTa architecture that demonstrates high performance and computational efficiency.

This work addresses the scarcity of specialized resources for Catalan NLP by first constructing a new, balanced corpus of over 23,000 samples. The primary CSXSC model is then trained on this dataset and benchmarked against two large-scale, 7-billion-parameter language models (`Starling-LM-7B-alpha` and `Salamandra-7B`). The findings demonstrate that for this classification task, the smaller, specialized encoder-only model not only achieves superior accuracy but is also over 24 times more efficient, highlighting the critical importance of architectural suitability.

---

## Project Workflow and Notebooks

The project is structured across three sequential Jupyter Notebooks, which cover the entire workflow from data creation to final model evaluation.

### 1. `dataset.ipynb` - Dataset Creation and Curation
This notebook details the comprehensive data engineering pipeline developed for this study. The primary objective was to construct a high-quality, balanced dataset for the three-class sentiment task (positive, negative, neutral). The process documented in this notebook includes:
* **Data Aggregation:** Combining three distinct source datasets: `GuiaCat`, `CaSSA`, and `GoEmotions`.
* **Rebalancing Strategy:** Implementing a sampling algorithm to augment the initial Catalan corpus with instances from GoEmotions to achieve a target distribution of approximately 40% positive, 30% negative, and 30% neutral.
* **Machine Translation:** Translating the sampled English texts from GoEmotions into Catalan using the high-quality Aina English-Catalan Translator.
* **Quality Curation:** A multi-stage quality control process, including an automated filtering step using the `Salamandra-7B-Instruct` model to score and remove low-quality translations, followed by a final manual review to clean noise and correct labels.

### 2. `train_model.ipynb` - Model Training and Hyperparameter Tuning
This notebook focuses on the training and optimization of the primary **CSXSC** model (`projecte-aina/rober-base-ca-v2`). The key activities in this notebook are:
* **Hyperparameter Search:** A systematic grid search is conducted to identify the optimal combination of hyperparameters (learning rate, batch size, weight decay, number of epochs).
* **Regularization:** Implementation of several techniques to prevent overfitting and improve generalization, including weight decay, learning rate scheduling with warmup, and an early stopping strategy based on the best validation set performance.
* **Performance Analysis:** A detailed, epoch-by-epoch analysis of the model's performance on the validation set to identify the optimal training checkpoint and diagnose the onset of overfitting.

### 3. `evaluate_model.ipynb` - Final Evaluation and Comparative Analysis
This notebook contains the final phase of the research, where the definitive performance of all models is assessed. The steps include:
* **Final CSXSC Evaluation:** Retraining the optimized CSXSC model on a combined corpus of the training and validation splits and evaluating its final performance on the held-out test set.
* **Large Model Fine-Tuning:** Fine-tuning the two 7-billion-parameter models (`Starling-LM-7B-alpha` and `Salamandra-7B`) using advanced, memory-efficient optimization techniques, primarily **QLoRA**.
* **Comparative Analysis:** A final evaluation of all three models on the test set, compiling their results on predictive metrics (Accuracy, F1, QWK) and computational efficiency (inference speed) to generate the final comparative analysis.

---

## Models and Datasets on Hugging Face

The primary assets developed in this project are publicly available on the Hugging Face Hub:

* **Dataset:** [`Danie1Arias/sentiment-analysis-catalan-reviews`](https://huggingface.co/datasets/Danie1Arias/sentiment-analysis-catalan-reviews)
    * The final, balanced dataset containing 23,788 Catalan texts for sentiment analysis, split into training, validation, and test sets.
* **Model:** [`Danie1Arias/csxsc`](https://huggingface.co/Danie1Arias/csxsc)
    * The final, fine-tuned CSXSC model, which achieved the best performance and efficiency in this study.

---

## Setup and Usage

To replicate the findings of this study, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Danie1Arias/CSXSC.git](https://github.com/Danie1Arias/CSXSC.git)
    cd CSXSC
    ```
2.  **Install dependencies:** It is recommended to create a virtual environment. Install the required Python packages using:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the notebooks:** The notebooks are designed to be run in sequential order:
    1.  `dataset.ipynb`
    2.  `train_model.ipynb`
    3.  `evaluate_model.ipynb`

---

## Citation

This repository contains the work for a Master's Thesis. If you use the code, dataset, or model in your research, please cite the original work:

```bibtex
@mastersthesis{arias2025csxsc,
  title={From Traditional to Large Language Models: A Novel NLP-Based Model for Sentiment Analysis in Social Media},
  author={Arias Cámara, Daniel},
  year={2025},
  school={Rovira i Virgili University}
}