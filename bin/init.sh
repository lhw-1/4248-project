# Setup data directory
mkdir data/
cd data/

# Download the PredictAndExplain model
mkdir PredictAndExplain/
cd PredictAndExplain/
gdown https://drive.google.com/u/0/uc?id=1w8UlNQ5yvZPNu4RgVkgICB6qsefkjolG
unzip eSNLI_PredictAndExplain.zip
rm eSNLI_PredictAndExplain.zip
cd ..

# # Download the ExplainThenPredictAttention model
# mkdir ExplainThenPredictAttention/
# cd ExplainThenPredictAttention/
# gdown https://drive.google.com/u/0/uc?id=1l7dnml7mDnT72QrwZMmA7VGIsWjVpQT6
# unzip eSNLI_ExplainThenPredictAttention.zip
# rm eSNLI_ExplainThenPredictAttention.zip
# cd ..

# # Download the ExplanationsToLabels model
# mkdir ExplanationsToLabels/
# cd ExplanationsToLabels/
# gdown https://drive.google.com/u/0/uc?id=1_rFGlFYHSJ1xqjA2lDjzBvO5mf7INo1A
# unzip eSNLI_expls_to_labels.zip
# rm eSNLI_expls_to_labels.zip
# cd ..

# Download the test dataset
#curl -o test.csv https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv

mkdir dataset
curl -o dataset/esnli_train_1.csv https://github.com/OanaMariaCamburu/e-SNLI/blob/master/dataset/esnli_train_1.csv
curl -o dataset/esnli_train_2.csv https://github.com/OanaMariaCamburu/e-SNLI/blob/master/dataset/esnli_train_2.csv
curl -o dataset/esnli_dev.csv https://github.com/OanaMariaCamburu/e-SNLI/blob/master/dataset/esnli_dev.csv
curl -o dataset/esnli_test.csv https://github.com/OanaMariaCamburu/e-SNLI/blob/master/dataset/esnli_test.csv
mkdir dataset/GloVe
curl -o dataset/GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip dataset/GloVe/glove.840B.300d.zip -d dataset/GloVe/
rm -r dataset/GloVe/glove.840B.300d.zip

#download infersent encoders
#infersent1.pkl
gdown 1csv3pP-tikFZHWLEVLDMqSVDnPx77AQz
#intersent2.pkl
gdown 1NaB79RJEW79VLHo0WBCR1SxLwbsTUUd5

#download trained InferSent model
gdown 1peT6jGH63zzYroqLpWo_eGZBE5E3PnGM

# Cleanup
cd ..