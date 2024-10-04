# SVHN Digit Classification Using CNN

## 1. Introduction
The objective of this project is to build a Convolutional Neural Network (CNN) for classifying images of digits from the Street View House Numbers (SVHN) dataset. The SVHN dataset is composed of real-world digit images obtained from house numbers in Google Street View images, which poses a challenging digit recognition task due to noise, occlusions, and varying illumination conditions. This project focuses on achieving accurate digit classification using a deep learning model built with TensorFlow/Keras.

The task is a multi-class classification problem, where the goal is to assign one of 10 classes (digits 0-9) to each image. The model's performance is evaluated on a test set, and accuracy is used as the primary evaluation metric.

## 2. Dataset Overview
The SVHN dataset used in this project consists of two parts:
- **Training Set**: 73,257 color images (32x32 pixels) of digits.
- **Test Set**: 26,032 color images (32x32 pixels) of digits.

The images were labeled with corresponding digits (0-9), with '10' representing the digit '0'.

Steps taken to process the data:
- Loaded the data using `scipy.io.loadmat`.
- Reshaped the data to move the number of samples to the first dimension (from `(32, 32, 3, N)` to `(N, 32, 32, 3)`).
- Normalized pixel values from the range [0, 255] to [0, 1].
- Converted the labels to one-hot encoded vectors for multi-class classification.

## 3. Data Preprocessing
### 3.1 Loading the Dataset
The dataset files are in `.mat` format, which we load using the `loadmat()` function from the SciPy library. The data for each set consists of two components:
- **X**: The images in shape (width, height, channels, samples).
- **y**: The corresponding labels for each image.

We reshape the data using `np.transpose()` to match the Keras input format of `(samples, width, height, channels)`.

### 3.2 Reshaping and Normalizing the Data
The original image pixel values range from 0 to 255. To make the training process more efficient and stable, we normalize the pixel values to fall within the range of 0 and 1.

### 3.3 Handling Labels and One-Hot Encoding
The labels in the dataset represent the digit '0' as '10'. To ensure correct classification, we map all occurrences of '10' to '0'. Then, we apply one-hot encoding to the labels using the `to_categorical()` function, which is essential for the multi-class classification task.

### 3.4 Splitting the Training Set
To validate the model during training, we split 20% of the training data into a validation set using `train_test_split()`.

## 4. CNN Model Architecture
A CNN was implemented using the Keras Sequential API, which allows for stacking layers sequentially. The model includes several convolutional layers, pooling layers, and dense layers to capture the spatial features of digit images.

### 4.1 Model Structure
- **Input Layer**: Accepts images in a 32x32 RGB format.
- **Convolutional Layers**: Three layers of convolution with increasing filter sizes (32, 64, 128). Each layer is followed by **MaxPooling** to downsample the feature maps.
- **Flattening Layer**: Converts the 3D output from the convolutional layers into a 1D vector for the dense layers.
- **Fully Connected Layer**: A dense layer with 128 neurons followed by **Dropout** to reduce overfitting.
- **Output Layer**: A dense layer with 10 neurons (corresponding to 10 digits) using **softmax activation** for classification.

### 4.2 Compiling the Model
The model was compiled using the **Adam optimizer**, which is effective for deep learning tasks, and the **categorical cross-entropy** loss function, which is appropriate for multi-class classification.

## 5. Model Training
The model was trained for 20 epochs with a batch size of 64. During training, both the training and validation loss/accuracy were monitored to track the performance of the model.

## 6. Model Evaluation and Results
### 6.1 Test Evaluation
After training, the model was evaluated on the test dataset. The final test accuracy achieved was **91.4%**, which indicates that the model generalizes well to unseen data.

### 6.2 Plotting the Training History
We plotted the training and validation accuracy and loss across epochs to visually analyze the model’s performance.

The plots show steady improvement in both training and validation accuracy, while the loss decreases consistently, confirming that the model is learning effectively.

## 7. Conclusion
The CNN model achieved strong performance on the SVHN dataset with a test accuracy of **91.4%**. The use of convolutional layers allowed the model to effectively capture and learn the spatial relationships between pixels, making it capable of distinguishing between digits even in challenging real-world images.

### 7.1 Challenges
Some challenges faced in this project included:
- Correctly handling the label '10' to represent the digit '0'.
- Balancing the training process to avoid overfitting, which was mitigated by using a Dropout layer.
