# Pytorch-Transformers-Classification


This repository is based on the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library by HuggingFace. It is intended as a starting point for anyone who wishes to use Transformer models in text classification tasks.

Please refer to this [Medium article](https://medium.com/p/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca?source=email-6b1e2355088e--writer.postDistributed&sk=f21ffeb66c03a9804572d7063f57c04e) for further information on how this project works.

Table of contents
=================

<!--ts-->
   * [Setup](#Setup)
      * [*Very* quickstart](#very-quickstart)
      * [With Conda](#with-conda)
   * [Usage](#usage)
      * [Yelp Demo](#yelp-demo)
      * [Custom Datasets](#custom-datasets)
      * [Current Pretrained Models](#current-pretrained-models)
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

### Current Pretrained Models

The table below shows the currently available model types and their models. You can use any of these by setting the `model_type` and `model_name` in the `args` dictionary. For more information about pretrained models, see [HuggingFace docs](https://huggingface.co/pytorch-transformers/pretrained_models.html).

| Architecture        | Model Type           | Model Name  | Details  |
| :------------- |:----------| :-------------| :-----------------------------|
| BERT      | bert | bert-base-uncased | 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on lower-cased English text. |
| BERT      | bert | bert-large-uncased | 24-layer, 1024-hidden, 16-heads, 340M parameters.<br>Trained on lower-cased English text. |
| BERT      | bert | bert-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on cased English text. |
| BERT      | bert | bert-large-cased | 24-layer, 1024-hidden, 16-heads, 340M parameters.<br>Trained on cased English text. |
| BERT      | bert | bert-base-multilingual-uncased | (Original, not recommended) 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on lower-cased text in the top 102 languages with the largest Wikipedias |
| BERT      | bert | bert-base-multilingual-cased | (New, recommended) 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on cased text in the top 104 languages with the largest Wikipedias |
| BERT      | bert | bert-base-chinese | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on cased Chinese Simplified and Traditional text. |
| BERT      | bert | bert-base-german-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>Trained on cased German text by Deepset.ai |
| BERT      | bert | bert-large-uncased-whole-word-masking | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>Trained on lower-cased English text using Whole-Word-Masking |
| BERT      | bert | bert-large-cased-whole-word-masking | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>Trained on cased English text using Whole-Word-Masking |
| BERT      | bert | bert-large-uncased-whole-word-masking-finetuned-squad | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>The bert-large-uncased-whole-word-masking model fine-tuned on SQuAD |
| BERT      | bert | bert-large-cased-whole-word-masking-finetuned-squad | 24-layer, 1024-hidden, 16-heads, 340M parameters <br>The bert-large-cased-whole-word-masking model fine-tuned on SQuAD |
| BERT      | bert | bert-base-cased-finetuned-mrpc | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>The bert-base-cased model fine-tuned on MRPC |
| XLNet      | xlnet | xlnet-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>XLNet English model |
| XLNet      | xlnet | xlnet-large-cased | 24-layer, 1024-hidden, 16-heads, 340M parameters. <br>XLNet Large English model |
| XLM      | xlm | xlm-mlm-en-2048 | 12-layer, 2048-hidden, 16-heads <br>XLM English model |
| XLM      | xlm | xlm-mlm-ende-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-German Multi-language model |
| XLM      | xlm | xlm-mlm-enfr-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-French Multi-language model |
| XLM      | xlm | xlm-mlm-enro-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-Romanian Multi-language model |
| XLM      | xlm | xlm-mlm-xnli15-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM Model pre-trained with MLM on the 15 XNLI languages |
| XLM      | xlm | xlm-mlm-tlm-xnli15-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM Model pre-trained with MLM + TLM on the 15 XNLI languages |
| XLM      | xlm | xlm-clm-enfr-1024 | 12-layer, 1024-hidden, 8-heads <br>XLM English model trained with CLM (Causal Language Modeling) |
| XLM      | xlm | xlm-clm-ende-1024 | 6-layer, 1024-hidden, 8-heads <br>XLM English-German Multi-language model trained with CLM (Causal Language Modeling) |
| RoBERTa      | roberta | roberta-base | 125M parameters <br>RoBERTa using the BERT-base architecture |
| RoBERTa      | roberta | roberta-large | 24-layer, 1024-hidden, 16-heads, 355M parameters <br>RoBERTa using the BERT-large architecture |
| RoBERTa      | roberta | roberta-large-mnli | 24-layer, 1024-hidden, 16-heads, 355M parameters <br>roberta-large fine-tuned on MNLI. |

### Custom Datasets

When working with your own datasets, you can create a script/notebook similar to [data_prep.ipynb](data_prep.ipynb) that will convert the dataset to a Pytorch-Transformer ready format.

The data needs to be in `tsv` format, with four columns, and no header.

This is the required structure.

- `guid`: An ID for the row.
- `label`: The label for the row (should be an int).
- `alpha`: A column of the same letter for all rows. Not used in classification but still expected by the `DataProcessor`.
- `text`: The sentence or sequence of text.

### Evaluation Metrics

The evaluation process in the [run_model.ipynb](run_model.ipynb) notebook outputs the confusion matrix, and the Matthews correlation coefficient. If you wish to add any more evaluation metrics, simply edit the `get_eval_reports()` function in the notebook. This function takes the predictions and the ground truth labels as parameters, therefore you can add any custom metrics calculations to the function as required.

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Pytorch-Transformers](https://github.com/huggingface/pytorch-transformers) library.
