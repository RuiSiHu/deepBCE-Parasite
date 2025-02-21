# deepBCE-Parasite

# 1 Description


In this study, we developed a novel prediction tool, named deepBCE-Parasite, designed to identify B-cell epitopes in parasitic organisms, including protozoan and helminth parasites, utilizing both deep learning and traditional machine learning approaches. In the deep learning framework, the deepBCE-Parasite model processes input amino acid sequences through embedding layers, positional encoding, multi-head self-attention mechanisms, and a feature optimization module, culminating in classification via fully connected layers. For the traditional machine learning approach, we employed twelve feature representations and used recursive feature elimination (RFE) for feature selection. Furthermore, four machine learning algorithms, including Random Forest (RF), LightGBM (LGBM), Gaussian Naive Bayes (GNB), and Support Vector Machine (SVM), were applied to construct the deepBCE model. Additionally, we predicted B-cell epitopes in *Fasciola hepatica* based on large-scale proteomics data, and discussed the related pathogenic mechanisms and their implications in vaccine development.


# 2 Requirements

It is recommended to use either Anaconda or Miniconda to install Python, with the specific version being Python 3.8.19. To create a conda virtual environment, follow the command below:

    conda create -n deepBCE python==3.8.19

Activate the newly created environment with the following command:

    source activate deepBCE

Within this Python environment, ensure the following packages and their corresponding versions are installed using pip:

pip install torch==2.4.1
pip install numpy==1.24.4
pip install pandas==2.0.3
pip install openpyxl==3.1.5
pip install scikit-learn==1.0
pip install lightgbm==4.5.0


# 3 Running

Navigate to the deepBCE-Parasite directory and run the following command:

python deepBCE-DL.py -i test1.fasta -o Deep_test1_results.csv

    python deepBCE-DL.py -h

Sequence-Based Prediction of B-cell Epitope in Human and Veterinary Parasites Using Transformer-based Deep Learning

Optional arguments:

     -h: --help: displays help information and exits
     -i: input file in FASTA format
     -o: output file name (optional)

This command employs deep learning method to predict B-Cell epitopes in parasitic organisms. The prediction results will be saved in the newly created DeepResults folder.

For further validation, traditional machine learning algorithms can be applied to refine the predictions. Use the following command:

    python BCE-ML.py -h

Sequence-Based Prediction of B-Cell Epitopes in Human and Veterinary Parasites Using Feature Representation Learning

Optional arguments:

    -h: --help: displays help information and exits
    -i: input file in FASTA format
    -f: {AAC, ASDC, CKSAAGP, CKSAAP, DDE, DPC, GAAC, GDPC, GTPC, PAAC, QSOrder, SOCNumber}: feature type
    -c: {GNB, LGBM, RF, SVM}: classifier type

The -f parameter allows to select a specific feature type, while -c specifies the machine learning algorithm to be used.

Due to GitHub's storage limitations, the machine learning models can be downloaded from https://huggingface.co/huruisi and placed in the Models directory.

