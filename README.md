## <ins>**Overview**</ins>

In this assignment, we are required to train a model which parses chess board images, for the purpose of detecting and analyzing the position of chess pieces on the corresponding board. We are given a preset data folder consisting of **250 chess board images** (125 clean and noisy images each, given in `./data/clean/capablanca` and `./data/noisy/capablanca/` respectively). The **first 100 images** (`001.jpg - 100.jpg`) in each folder are used as **data training labels** (represented in `boards.train.json`, used in `train.py`); whereas the **remaining 25 images** (`101.jpg - 125.jpg`) are used for **evaluation and testing** (represented in `boards.dev.json`, used in `evaluate.py`). Each image is **400x400 pixels** and each square is **50x50 pixels**, forming a 8x8 square chess board.

We are only allowed to modify `system.py`, in which the functions we're required to complete are **`classify`**, **`reduce_dimensions`**, **`process_training_data`**, **`classify_squares`** and **`classify_board`**. In addition, we are limited to **10 feature vectors** (**`N_DIMENSIONS = 10`**) for our classification system, and we are only allowed to use the default Python libraries alongside **numpy** and **scipy** packages.

As the given code already handles the file handling and initial processing steps (e.g. image input via **`load_square_images`** function in `utils.py` and conversion to feature vectors via **`image_to_feature_vectors`** function in `system.py`), we can proceed directly to the implementation of **`process_training_data`**.

## **<ins>Functions</ins>**

### <ins>1\. process\_training\_data</ins>

Using the default parameters (**`fvectors_train`** and **`labels_train`**), we can initialize a model dictionary containing the feature vectors and labels used for training. After running the feature vectors through the **reduce\_dimensions** function, we can update the model dictionary and return it for use in **train()**.

### <ins>2\. reduce\_dimensions</ins>

As suggested in the assignment brief, I have decided to implement the **PCA (Principal Component Analysis)** unsupervised learning method for dimensionality reduction. I implemented my own PCA method mathematically using NumPy based on the formulae and steps detailed on the Wikipedia page regarding the method.

First, we have to calculate the empirical mean of the dataset, followed by the covariance matrix using the transposed data array. Then, we derive the eigenvalues and eigenvectors using the covariance matrix and sort the eigenvectors in terms of descending eigenvalues. Next, we compute the principal components from the sorted eigenvectors up to N\_DIMENSIONS, and finally obtain the reduced dataset from the dot product between the matrices of the centred data array and principal components respectively.

Since the variance of principal components is cumulative, with each subsequent component always having less variance than the previous one (resembling an inverse-square graph), it is generally recommended to find an optimal amount of feature vector dimensions as a cut-off point (typically at **90-99% cumulative variance**). This effectively reduces the training time and processing power required, and negates the curse of dimensionality. However, as we are limited to only 10 dimensions in this assignment, we can infer hypothetically that it is nowhere near the 99% variance cut-off mark, hence I opted to pass the maximum amount of 10 feature vectors (`N_DIMENSIONS = 10`) into the reduce\_dimensions function, in order to obtain a score with the highest accuracy.

During my research, I have come across another algorithm/model known as **LDA (Linear Discriminant Analysis)** which claims to have higher classification accuracy compared to PCA, as it is a **supervised method** of data training (unlike **PCA which is unsupervised**). Unfortunately, it is significantly harder to implement in Python without the scikit-learn package and most likely will have relatively lower efficiency, therefore I decided to implement PCA as it is more suited towards the context of this assignment.

**Note:** Since **`reduce_dimensions`** is called in both **`process_training_data`** and **`evaluate`** functions, an if-else block is required to prevent the model dictionary of existing trained data from being overwritten during the evaluation phase.

### <ins>3\. classify</ins>

Once the training feature vectors have been reduced, we can apply labels to each test feature vector based on the **k-Nearest Neighbour (k-NN)** algorithm. In order to accomplish this, we iterate through each test feature vector and calculate the shortest **Euclidean distance** to its surrounding feature vectors. Thus, the nearest training feature vector is selected and the test feature vector is labeled accordingly (e.g. R for white rook) and returned as a string array representing each square in all 25 chess images used for testing (**25 \* 64 = 1600** values in the array). This function is then called in the **`classify_squares`** and `classify_boards` functions, and the labeling accuracy is calculated by comparing the returned string array to the pre-existing data in `boards.dev.json`. 

## <ins>**External References**</ins>

1.  **[Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) - Wikipedia**
2.  **[Data Analysis 6: Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=TJdH6rPA-TI "Data Analysis 6: Principal Component Analysis (PCA) - Computerphile") - Computerphile (YouTube)**
3.  **[Machine Learning Tutorial Python - 19: Principal Component Analysis (PCA) with Python Code](https://www.youtube.com/watch?v=8klqIM9UvAc "Machine Learning Tutorial Python - 19: Principal Component Analysis (PCA) with Python Code") - codebasics (YouTube)**
4.  **[PCA : the math - step-by-step with a simple example](https://www.youtube.com/watch?v=S51bTyIwxFs "PCA : the math - step-by-step with a simple example") - TileStats (YouTube)**
5.  **[Scipy Documentation](https://docs.scipy.org/doc/scipy/reference/main_namespace.html)**