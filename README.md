# Pytorch-Transformers-Classification


This repository is based on the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library by HuggingFace. It is intended as a starting point for anyone who wishes to use Transformer models in text classification tasks.

Please refer to this [Medium article](https://medium.com/@chaturangarajapakshe/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca?source=friends_link&sk=f21ffeb66c03a9804572d7063f57c04e) for further information on how this project works.

Table of contents
=================

<!--ts-->
   * [Setup](#Setup)
      * [*Very* quickstart](#cery-quickstart)
      * [With Conda](#with-conda)
   * [Usage](#usage)
      * [Yelp Demo](#yelp-demo)
      * [Custom Datasets](#custom-datasets)
      * [Evaluation Metrics](#evaluation-metrics)
   * [Acknowledgements](#acknowledgements)
<!--te-->

## Setup

### *Very* quickstart

Try this [Google Colab Notebook](colab_quickstart.ipynb) for a quick preview. You can run all cells without any modifications to see how everything works. However, due to the 12 hour time limit on Colab instances, the dataset has been undersampled from 500 000 samples to about 5000 samples. For such a tiny sample size, everything should complete in about 10 minutes.

### With Conda

1. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/distribution/)
2. Create a new virtual environment and install packages.  
`conda create -n transformers python pandas tqdm jupyter`  
`conda activate transformers`  
If using cuda:  
  `conda install pytorch cudatoolkit=10.0 -c pytorch`  
else:  
  `conda install pytorch cpuonly -c pytorch`  
`conda install -c anaconda scipy`  
`conda install -c anaconda scikit-learn`  
`pip install pytorch-transformers`  
3. Clone repo.
`git clone https://github.com/ThilinaRajapakse/pytorch-transformers-classification.git`

## Usage

### Yelp Demo

This demonstration uses the Yelp Reviews dataset.

Linux users can execute [data_download.sh](data_download.sh) to download and set up the data files.

If you are doing it manually;

1. Download [Yelp Reviews Dataset](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz).
2. Extract `train.csv` and `test.csv` and place them in the directory `data/`.

Once the download is complete, you can run the [data_prep.ipynb](data_prep.ipynb) notebook to get the data ready for training.

Finally, you can run the [run_model.ipynb](run_model.ipynb) notebook to fine-tune a Transformer model on the Yelp Dataset and evaluate the results.

### Custom Datasets

When working with your own datasets, you can create a script/notebook similar to [data_prep.ipynb](data_prep.ipynb) that will convert the dataset to a Pytorch-Transformer ready format.

The data needs to be in `tsv` format, with four columns, and no header.

This is the required structure.

guid: An ID for the row.
label: The label for the row (should be an int).
alpha: A column of the same letter for all rows. Not used in classification but still expected by the `DataProcessor`.
text: The sentence or sequence of text.

### Evaluation Metrics

The evaluation process in the [run_model.ipynb](run_model.ipynb) notebook outputs the confusion matrix, and the Matthews correlation coefficient. If you wish to add any more evaluation metrics, simply edit the `get_eval_reports()` function in the notebook. This function takes the predictions and the ground truth labels as parameters, therefore you can add any custom metrics calculations to the function as required.

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library.
