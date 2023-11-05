## <ins>**Overview**</ins>

In this assignment, we are required to train a model which parses chess board images, for the purpose of detecting and analyzing the position of chess pieces on the corresponding board. We are given a preset data folder consisting of **250 chess board images** (125 clean and noisy images each, given in **`./data/clean/capablanca`** and `./data/noisy/capablanca/` respectively). The **first 100 images** (**`001.jpg - 100.jpg`**) in each folder are used as **data training labels** (represented in `boards.train.json`, used in **`train.py`**); whereas the **remaining 25 images** (**`101.jpg - 125.jpg`**) are used for **evaluation and testing** (represented in **`boards.dev.json`**, used in **`evaluate.py`**). Each image is **400x400 pixels** and each square is **50x50 pixels**, forming a 8x8 square chess board.

We are only allowed to modify **`system.py`**, in which the functions we're required to complete are `classify, reduce_dimensions`, **`process_training_data`**, **`classify_squares`** and **`classify_board`**. In addition, we are limited to **10 feature vectors** (**`N_DIMENSIONS = 10`**) for our classification system, and we are only allowed to use the default Python libraries alongside **numpy** and **scipy** packages.

As the given code already handles the file handling and initial processing steps (e.g. image input via **`load_square_images`** function in utils.py and conversion to feature vectors via **`image_to_feature_vectors`** function in system.py), we can proceed directly to the implementation of **`process_training_data`**.

## **<ins>Functions</ins>**

### <ins>1\. process\_training\_data</ins>

Using the default parameters (**`fvectors_train`** and **`labels_train`**), we can initialize a model dictionary containing the feature vectors and labels used for training. After running the feature vectors through the **reduce\_dimensions** function, we can update the model dictionary and return it for use in **train()**.