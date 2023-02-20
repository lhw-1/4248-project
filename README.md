# 4248-project

## Setup

Make sure that you have Python version 3 or above before running the scripts.

To install the dependencies, run the following command:

```
pip install -r requirements.txt
```

Next, to download the models, run either `sh bin/init.sh` or `bash bin/init.sh` depending on your system. This should download the model to be tested (currently only PredictAndExplain). Note that we are using `gdown` tool for downloading large files from Google Drive. This script will also download the test file from the [e-SNLI Repository](https://github.com/OanaMariaCamburu/e-SNLI), specifically [this file](https://github.com/OanaMariaCamburu/e-SNLI/blob/master/dataset/esnli_test.csv).

Links for the three trained models available can be found [here](https://github.com/OanaMariaCamburu/e-SNLI#trained-models).

Once the setup is complete, you should be able to run the code in the Notebook without issues :)
