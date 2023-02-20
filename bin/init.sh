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
curl -o test.csv https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv

# Cleanup
cd ..