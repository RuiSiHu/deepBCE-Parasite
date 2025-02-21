# deepBCE-Parasite

# 1 Description


In this study, we introduce deepBCE-Parasite, a novel prediction tool specifically designed to identify B-cell epitopes (BCEs) in human and veterinary parasitic organisms, encompassing both protozoan and helminth parasites. This tool integrates deep learning and traditional machine learning approaches to achieve robust performance. Within the deep learning framework, the deepBCE-Parasite model processes input amino acid sequences through a series of advanced computational layers, including embedding layers, positional encoding, multi-head self-attention mechanisms, and a feature optimization module, ultimately performing classification via fully connected layers. For the traditional machine learning approach, this work utilized twelve distinct feature representations and employed recursive feature elimination (RFE) to optimize feature selection. Additionally, this work implemented four machine learning algorithms—Random Forest (RF), LightGBM (LGBM), Gaussian Naive Bayes (GNB), and Support Vector Machine (SVM)—to construct the deepBCE model. deepBCE-Parasite exhibits strong performance in predicting BCEs for diverse parasitic pathogens, providing a powerful tool for advancing epitope-based vaccines, antibodies, and diagnostics in parasitology.

# 2 Requirements

It is recommended to use either Anaconda or Miniconda to install Python, with the specific version being Python 3.8.19. To create a conda virtual environment, please follow the command below:

    conda create -n deepBCE python==3.8.19

Activate the newly created environment with the following command:

    source activate deepBCE

Within this Python environment, please ensure the following packages and their corresponding versions are installed using pip:

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
    -f: (AAC, ASDC, CKSAAGP, CKSAAP, DDE, DPC, GAAC, GDPC, GTPC, PAAC, QSOrder, SOCNumber): feature type
    -c: (GNB, LGBM, RF, SVM): classifier type

The -f parameter allows to select a specific feature type, while -c specifies the machine learning algorithm to be used.

Due to GitHub's storage limitations, the machine learning models can be downloaded from https://huggingface.co/huruisi and placed in the Models directory.

