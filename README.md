# DS-4002-Project-3
## What Tree is This - Using ResNet18 as a Tree Image Classifier
This project trains two ResNet18 models on tree images, one from scratch and one pretrained. In the end, we arrived at one final model that had the greatest accuracy at **___TO-DO___**, which was the **___TO-DO___** model.

## Contents
* **`/SCRIPTS`**: Reusable code for data pulls, cleaning, modeling, plotting.
* **`/DATA`**: All data states, including intial and final.
* **`/OUTPUT`**: Final figures and metrics used in the write-up, as well as figures from MI2.
* **`LICENSE`**: An MIT license for our project
* **`requirements.txt`**: Necessary package installations
* **`.gitignore`**: Ignore local python virtual environment

## Software and Platform
**Software stack**

* **Language:** Python 3.10+ (tested on 3.11)
* **Environment:** VS Code
* **Package manager:** `pip` (Conda optional)

**Key Python packages**

* Standard Library: **___TO-DO___**
* External: **___TO-DO___**
> Install all via `pip install -r requirements.txt`.

**Platform used for development**

* **macOS** (Apple Silicon/Intel)
  The project also works on Windows with the same Python packages.

## Documentation Map
**___TO-DO___**
Below is the project's folder structure.

```
DS-4002-Project1-GPT-6.0
├── DATA/                                   : includes all data
│   ├── Final/                                  : final data
│   │   ├── full_data/                               : all data together as images and labels
│   │   │   ├── train/                                  : training split images and labels
│   │   │   ├── validate/                               : validating split images and labels
│   │   │   └── test/                                   : testing split images and labels
│   │   └── sample_data/                                 : all data put into each split
│   │       ├── train/                                  : training split images and labels
│   │       ├── validate/                               : validating split images and labels
│   │       └── test/                                   : testing split images and labels
│   ├── Initial/                                : initial data
│   │   ├── full_data/                              : full data
│   │       ├── ______/
│   │       ├── ______/
│   │       └── ______/
│   │   └── sample_data/                            : sample data
│   │       ├── ______/
│   │       ├── ______/
│   │       └── ______/
|   └── README                                  : Metadata explanation
├── OUTPUT/                                     : includes final outputs from trained models
│   ├── Final/                                      : -
│   │   └── models/                                     : both generated ResNet18 Models
│   └── M12/                                        : EDA for MI2 shown in DATA/README.md
│       ├── sample_train_split_images_per_class.png     : EDA chart that counts number of classes per training split from the sample data
│       └── sample_train_test_images_per_class.png      : EDA chart that counts number of classes per test split from the sample data
├── SCRIPTS/                                    : folder holding all scripts
│   ├── tbd                                         : tbd
│   ├── tbd                                         : tbd
│   └── tbd                                         : tbd
├── LICENSE.md                                  : general file - MIT licensing
├── requirements.txt                            : general file - contains necessary packages
├── .gitignore                                  : general file - keeps certain files off of github
└── .venv                                       : general file - private environment specific to a user
```
OUTPUT/MI2/sample_train_split_images_per_class.png
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
        - Windows --> `python -m venv .venv && .venv\Scripts\activate`
            - Note, if you using a newer python it may be `python3 ...` for all commands
    - Install required packages
        - within the terminal, `pip install -r requirements.txt`
    - Register the Jupyter kernel (if using notebooks outside VS Code):
        - `pip install ipykernel`
        - `python -m ipykernel install --user --name=.venv`
        - alternatively, in VS code you may have to define your interpreter as the one located in your venv. 
3) Run
    - **___TO-DO___**

### References
- [1] https://datasetninja.com/urban-street-tree-classification, “Urban Street: Tree Classification - Dataset Ninja,” Dataset Ninja, 2022. http://datasetninja.com/urban-street-tree-classification#download (accessed Nov. 05, 2025).
- [2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv.org, Dec. 10, 2015. https://arxiv.org/abs/1512.03385 
- [3]P. Baheti, “Train, Validation, and Test Set: How to Split Your Machine Learning Data,” V7labs.com, Sep. 13, 2021. https://www.v7labs.com/blog/train-validation-test-set
- [4]scikit learn, “3.1. Cross-validation: Evaluating Estimator Performance — scikit-learn 0.21.3 Documentation,” Scikit-learn.org, 2009. https://scikit-learn.org/stable/modules/cross_validation.html
- [5] GeeksforGeeks. 2025. “Evaluation Metrics in Machine Learning.” GeeksforGeeks. July 15, 2025. https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/.
- [6] “Using UVA’s High-Performance Computing Systems | Research Computing,” Virginia.edu, Dec. 03, 2024. https://www.rc.virginia.edu/userinfo/hpc/
- [7] C. Writer, “What Is ResNet-18? How to Use the Lightweight CNN Model,” Roboflow Blog, Jun. 23, 2025. https://blog.roboflow.com/resnet-18/
