# <ins>**COM2004 Assignment Development Journal**</ins>

## <ins>**Overview**</ins>

In this assignment, we are required to train a model which parses chess board images, for the purpose of detecting and analyzing the position of chess pieces on the corresponding board. We are given a preset data folder consisting of **250 chess board images** (125 clean and noisy images each, given in `./data/clean/capablanca` and `./data/noisy/capablanca/` respectively). The **first 100 images** (`001.jpg - 100.jpg`) in each folder are used as **data training labels** (represented in `boards.train.json`, used in `train.py`); whereas the **remaining 25 images** (`101.jpg - 125.jpg`) are used for **evaluation and testing** (represented in `boards.dev.json`, used in `evaluate.py`). Each image is **400x400 pixels** and each square is **50x50 pixels**, forming a 8x8 square chess board.

We are only allowed to modify `system.py`, in which the functions we're required to complete are **`classify`**, **`reduce_dimensions`**, **`process_training_data`**, **`classify_squares`** and **`classify_boards`**. In addition, we are limited to **10 feature vectors** (**`N_DIMENSIONS = 10`**) for our classification system, and we are only allowed to use the default Python libraries alongside **numpy** and **scipy** packages.

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

Once the training feature vectors have been reduced, we can apply labels to each test feature vector based on the **k-Nearest Neighbour (k-NN)** algorithm. In order to accomplish this, we iterate through each test feature vector and calculate the shortest **Euclidean distance** to its surrounding feature vectors. Thus, the nearest training feature vector is selected and the test feature vector is labeled accordingly (e.g. R for white rook) and returned as a string array representing each square in all 25 chess images used for testing (**25 \* 64 = 1600** values in the array). This function is then called in the **`classify_squares`** and **`classify_boards`** functions, and the labeling accuracy is calculated by comparing the returned string array to the pre-existing data in `boards.dev.json`.

In the second version, I have improved upon the k-NN algorithm used by taking the nearest 5 neighbouring training feature vectors into consideration, as represented by the **N\_NEIGHBOURS constant**. Upon further testing, I have found that **N\_NEIGHBOURS = 5** gives the **highest accuracy percentage** among N\_NEIGHBOURS = 1 to 10, hence I have finalized my code on this value. Additionally, I have used the **Counter** instance from Python's default **collections** library to determine the most common neighbouring training feature vector, in order to select the final k-NN value based on the majority.

### <ins>4\. classify\_squares</ins>

Using the default code provided. Obtains trained data from the model dictionary and passes it to the classify function for parsing.

### <ins>5\. classify\_boards</ins>

This is a fairly challenging function to implement, as we are required to consider each chess board as a whole and improve upon the accuracy (which is already quite optimized) without relying on algorithms and rather logical inference from chess gameplay rules. In extension to the classify\_squares function, we are able to use the generated label\_list as further information as we are essentially given the predicted positions of each chess piece on each chess board image. To separate the chess boards clearly, I have divided the label\_list array by 25 such that each nested array contains 64 elements corresponding to the chess pieces.

For my first attempt, I first tried to tabulate overall data regarding incorrect labels by modifying the evaluate function in evaluate.py, and the results are as shown in the table below, in descending order.

### <ins>**Incorrect Label Frequency Table (Clean data)**</ins>

| Wrong Label | Actual Label | Frequency |
| --- | --- | --- |
| P   | B   | 5   |
| n   | q   | 2   |
| N   | R   | 2   |
| K   | q   | 2   |
| R   | P   | 2   |
| K   | Q   | 2   |
| q   | Q   | 2   |
| k   | Q   | 2   |
| B   | P   | 1   |
| N   | P   | 1   |
| q   | b   | 1   |
| b   | q   | 1   |
| K   | B   | 1   |
| r   | b   | 1   |
| k   | K   | 1   |
| b   | K   | 1   |
| R   | N   | 1   |
| q   | n   | 1   |

### <ins>**Incorrect Label Frequency Table (Noisy data)**</ins>

| Wrong Label | Actual Label | Frequency |
| --- | --- | --- |
| R   | P   | 7   |
| N   | R   | 6   |
| R   | B   | 4   |
| K   | q   | 4   |
| .   | N   | 3   |
| P   | B   | 3   |
| K   | Q   | 3   |
| K   | R   | 3   |
| N   | Q   | 3   |
| B   | R   | 3   |
| N   | P   | 3   |
| R   | Q   | 2   |
| .   | B   | 2   |
| b   | r   | 2   |
| b   | q   | 2   |
| q   | b   | 2   |
| k   | n   | 2   |
| P   | p   | 2   |
| Q   | K   | 2   |
| r   | n   | 2   |
| k   | K   | 2   |
| k   | Q   | 2   |
| Q   | N   | 2   |
| r   | q   | 2   |
| K   | P   | 1   |
| n   | k   | 1   |
| .   | Q   | 1   |
| k   | r   | 1   |
| r   | B   | 1   |
| p   | b   | 1   |
| p   | P   | 1   |
| q   | k   | 1   |
| n   | q   | 1   |
| r   | k   | 1   |
| B   | P   | 1   |
| k   | q   | 1   |
| K   | .   | 1   |
| r   | R   | 1   |
| N   | .   | 1   |
| K   | r   | 1   |
| R   | N   | 1   |
| q   | Q   | 1   |
| q   | n   | 1   |
| q   | N   | 1   |

Given this data, I have identified the chess pieces with the highest frequency of incorrect labels (particularly those with an incorrect frequency of over 3), and implemented corresponding guard statements to update incorrect labels to the actual labels, if the normal threshold of chess piece types & color possible on a chess board is exceeded. This gives an additional 0.1% accuracy for noisy data board mode so far - not a significant improvement, and I will further attempt this after completing other assignments on hand.  

## <ins>**External References**</ins>

1.  **[Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) - Wikipedia**
2.  **[Data Analysis 6: Principal Component Analysis (PCA)](https://www.youtube.com/watch?v=TJdH6rPA-TI "Data Analysis 6: Principal Component Analysis (PCA) - Computerphile") - Computerphile (YouTube)**
3.  **[Machine Learning Tutorial Python - 19: Principal Component Analysis (PCA) with Python Code](https://www.youtube.com/watch?v=8klqIM9UvAc "Machine Learning Tutorial Python - 19: Principal Component Analysis (PCA) with Python Code") - codebasics (YouTube)**
4.  **[PCA : the math - step-by-step with a simple example](https://www.youtube.com/watch?v=S51bTyIwxFs "PCA : the math - step-by-step with a simple example") - TileStats (YouTube)**
5.  **[Scipy Documentation](https://docs.scipy.org/doc/scipy/reference/main_namespace.html) - Official Scipy Documentation**
6.  **[Python Documentation - collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter) - Official Python Documentation**
7.  **[python - Find the most common element in a list - Stack Overflow](https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list) - Stack Overflow Forums**