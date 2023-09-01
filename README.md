# Extracting the Genotype of Mice Samples from Public Bioinformatics Database Using BioNER Techniques

The field of biomedical research has seen remarkable progress, leading to a significant in-
crease in biological data stored in public databases like NCBI. These databases hold the
potential for discovering new medicines and identifying potential risks to human health.
However, the data within these repositories is often disorganized. Metadata in the repositories is incomplete and inconsistent. Much of the valuable information is stored in plain text,
making it difficult for researchers to gain new insights. One important type of information
is genotypic data, which can provide valuable insights into the inheritance of traits, disease
susceptibility, and intricate molecular interactions within living organisms. This dissertation focuses on identifying genotypes mentioned in project descriptions. This is achieved
by creating a collection of descriptions that already contain known genotypic information.
These description-genotype pairs are then used to train various BERT models. The process
of identifying genotypes proved to be a challenging task. Among the different methods we
tried, curriculum learning using BioRED was the most successful. While the model’s accuracy is imperfect, it provides researchers with structured data that can be further verified
manually. In this dissertation, we aim to improve the extraction of genotypic information
from the complex and unorganized realm of bioinformatics databases.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing.

### Prerequisites

Make sure you have these Python packages installed. 

```
scikit-learn
pandas
torch==1.11.0+cu113
torchvision==0.12.0+cu113
torchaudio==0.11.0
transformers
tokenizers
tokenizers
Bio
xml
```

## Running the files in the right order

* [ModefiedDownload.py](ModefiedDownload.py) can be run to download data and generate two data files, ```bioprojects.csv``` and ```biosamples.csv```.

* [Dataset Structuring.ipynb](Dataset%Structuring.ipynb) can be run to merge the datasets and divide them into train, test and validation.

* [Tokenization for BERT.ipynb](Tokenization%for%BERT.ipynb) can be run to tokenize the data using the token matching method.

* [Offset Tokenization.ipynb](Offset%Tokenization.ipynb) can be run to tokenize the data using the position-matching method.

* ```ModelComparisionBioBert.py```, ```ModelComparisionNormalLabelling.py```, ```ModelComparisionOffsetLabelling.py```, ```ModelComparisionPubMed.py``` and ```BioRED_Curiculam_learning.py``` are used to run fine-tuning, training and evaluation on the models.

## Author

* **Dhruv Kumar Patwari**
* **Jake Lever** (Supervisor)
