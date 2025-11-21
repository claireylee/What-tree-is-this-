# DS-4002-Project-3
## What Tree is This - Using ResNet18 as a Tree Image Classifier
This project trains two ResNet18 models on tree images, one from scratch and one pretrained. In the end, we arrived at one final model that had the greatest accuracy at 92.2%, which was the pretrained model on ImageNet. The model without pretraining performed with 71.1% accuracy

## Contents
* **`/SCRIPTS`**: Reusable code for data pulls, cleaning, modeling, plotting.
* **`/DATA`**: All data states, including intial and final.
* **`/OUTPUT`**: Final figures and metrics used in the write-up, as well as figures from MI2.
* **`LICENSE`**: An MIT license for our project
* **`requirements.txt`**: Necessary package installations
* **`.gitignore`**: Ignore local python virtual environment + full files

## Software and Platform
**Software stack**

* **Language:** Python 3.10+ (tested on 3.11)
* **Environment:** VS Code
* **Package manager:** `pip` (Conda optional)

**Key Python packages**

* Standard Library: os, json, shutil, pathlib, collections, random, errno, typing, time
* External: numpy, PIL, matplotlib.pyplot, torch, torchvision, sklearn.metrics

> Install all via `pip install -r requirements.txt` once your virtual environment is active.

**Platform used for development**

* **macOS** (Apple Silicon/Intel)
  The project also works on Windows with the same Python packages.
* **Rivanna** (UVA Supercomputer)
  Although all the scripts will work on a local machine, they were ran on the Rivanna supercomputer. It is recommended to use a machine with lots of computation power.

## Documentation Map
Below is the project's folder structure.

```
DS-4002-Project1-GPT-6.0
├── DATA/                                   : includes all data
│   ├── Final/                                  : final data
│   │   ├── sample_data/                            : sample data
│   │   │   ├── test/                                   : 10% testing split
│   │   │   |   ├── ann/                                    : a folder that holds json file annotations that match images
│   │   │   |   └── img/                                    : a folder that holds images that match annotations
│   │   │   ├── train/                                  : 70% training split
│   │   │   |   ├── ann/                                    : a folder that holds json file annotations that match images
│   │   │   |   └── img/                                    : a folder that holds images that match annotations
│   │   │   └── validate/                               : 20% validation split
│   │   │       ├── ann/                                    : a folder that holds json file annotations that match images
│   │   │       └── img/                                    : a folder that holds images that match annotations
│   │   └── full_data/                              : full data
│   │       └── -                                       : mirrors sample_data, just more image and annotation files
│   ├── Initial/                                : initial data
│   │   ├── sample_data/                            : sample data
│   │   │   ├── test/                                   : original dataset testing split
│   │   │   |   ├── ann/                                    : a folder that holds json file annotations that match images
│   │   │   |   └── img/                                    : a folder that holds images that match annotations
│   │   │   ├── train/                                  : original dataset training split
│   │   │   |   ├── ann/                                    : a folder that holds json file annotations that match images
│   │   │   |   └── img/                                    : a folder that holds images that match annotations
│   │   │   └── val/                                    : original dataset validation split
│   │   │       ├── ann/                                    : a folder that holds json file annotations that match images
│   │   │       └── img/                                    : a folder that holds images that match annotations
│   │   └── full_data/                              : full data
│   │       ├── -                                       : mirrors sample_data, just more image and annotation files
│   │       └── -                                       : one additional file, training_output.txt that shows the console outputs with timing
|   └── README                                  : Metadata explanation
├── OUTPUT/                                     : includes final outputs from trained models
│   ├── Final/                                      : final outputs
│   │   ├── sample_data/                                : sample data
│   │   |   ├── ResNet18_Pretrained_confusion_matrix.png    : confusion matrix results of our ResNet18 Pretrained Model
│   │   |   ├── ResNet18_Pretrained.pth                     : the actual ResNet18 Pretrained Model being stored
│   │   |   ├── ResNet18_Untrained_confusion_matrix.png     : confusion matrix results of our ResNet18 Model without any pretraining
│   │   |   ├── ResNet18_Untrained.pth                      : the actual ResNet18 Model without any pretraining being stored
│   │   |   └── metrics.json                                : metrics accross both models in a simple json file, showing the labels, results, and paths
│   │   └── full_data/                                  : full data
│   │       └── -                                           : mirrors sample_data, just results based on using the full data available
│   └── M12/                                        : EDA for MI2 shown in DATA/README.md
│       ├── sample_train_split_images_per_class.png     : EDA chart that counts number of classes per training split from the sample data
│       └── sample_train_test_images_per_class.png      : EDA chart that counts number of classes per test split from the sample data
├── SCRIPTS/                                    : folder holding all scripts
│   ├── clean_json_annotations.py                   : takes exissting json annotation files and extracts relavent information
│   ├── create_stratified_splits.py                 : create train-validate-test splits
│   └── train_and_test_models.py                    : trains and tests ResNet18 models
├── LICENSE.md                                  : general file - MIT licensing
├── requirements.txt                            : general file - contains necessary packages
├── .gitignore                                  : general file - keeps certain files off of github
└── .venv                                       : general file - private environment specific to a user
```

## Replication Instructions
1) Base Installations
    - Install Python 3.10+ and Git
        - more information can be found at https://www.python.org/downloads/ and https://git-scm.com/downloads
    - clone this repository (use the link https://github.com/claireylee/What-tree-is-this-)
        - clone --> `git clone https://github.com/claireylee/What-tree-is-this-`
        - enter the folder --> `cd What-tree-is-this`
2) Project Installations
    - Create python environment
        - macOS/Linux --> `python -m venv .venv && source .venv/bin/activate`
        - Windows --> `python -m venv .venv` followed by `.\.venv\Scripts\Activate.ps1`
            - Note, if you using a newer python it may be `python3 ...` for all commands
    - Install required packages
        - within the terminal, `pip install -r requirements.txt`
    - Register the Jupyter kernel (if using notebooks outside VS Code):
        - `pip install ipykernel`
        - `python -m ipykernel install --user --name=.venv`
        - alternatively, in VS code you may have to define your interpreter as the one located in your venv. 
3) Run
    - If you want to recreate all final files, within the root directory, run in order...
        1) `clean_json_annotations.py` to re-create the cleaned annotations
        2) `create_stratified_splits.py` to re-create the train-validate-test splits
        3) `train_and_test_models.py` to re-create and re-score the models. Note that this file is resource intensive to run.
    - View the output in `/OUTPUT/FINAL/full_data/`, which includes the models and performance (json metric fiels and confusion matrices).
    - Note that the scripts are currently coded to run on the full data, but the sample data results were included. To obtain sampled data results, go into each file and change the input and output directories (located near the top of each .py file) from `full_data` to `sample_data`.

### References
- [1] https://datasetninja.com/urban-street-tree-classification, “Urban Street: Tree Classification - Dataset Ninja,” Dataset Ninja, 2022. http://datasetninja.com/urban-street-tree-classification#download (accessed Nov. 05, 2025).
- [2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv.org, Dec. 10, 2015. https://arxiv.org/abs/1512.03385 
- [3]P. Baheti, “Train, Validation, and Test Set: How to Split Your Machine Learning Data,” V7labs.com, Sep. 13, 2021. https://www.v7labs.com/blog/train-validation-test-set
- [4]scikit learn, “3.1. Cross-validation: Evaluating Estimator Performance — scikit-learn 0.21.3 Documentation,” Scikit-learn.org, 2009. https://scikit-learn.org/stable/modules/cross_validation.html
- [5] GeeksforGeeks. 2025. “Evaluation Metrics in Machine Learning.” GeeksforGeeks. July 15, 2025. https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/.
- [6] “Using UVA’s High-Performance Computing Systems | Research Computing,” Virginia.edu, Dec. 03, 2024. https://www.rc.virginia.edu/userinfo/hpc/
- [7] C. Writer, “What Is ResNet-18? How to Use the Lightweight CNN Model,” Roboflow Blog, Jun. 23, 2025. https://blog.roboflow.com/resnet-18/
