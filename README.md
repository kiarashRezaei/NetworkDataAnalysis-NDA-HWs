Sure! Below is the template for the README file based on the provided code:

# Network Data Analysis - Homeworks

## Table of Contents

- [Homework 1- Failure Management in Optical Networks](#Homework1-FailureManagementinOpticalNetworks)
- [Homework 2- Traffic Prediction](#Homework2-TrafficPrediction)
- [Homework 3- QoS Estimation](#Homework3-QoSEstimation)


## Homework 1- Failure Management in Optical Networks

(Failure-Cause Identification using Network Traffic Analysis)

This folder contains the code for performing failure-cause identification using network traffic analysis. The analysis is carried out on a dataset of network traffic data, considering only failure classes for scenarios B (Attenuation) and C (Filtering).

## Task 7a


In Task 7a, we focus on identifying the causes of failure in network communication. We repeat tasks 6a) and 6b), but this time, we consider only failure classes for scenarios B (Attenuation) and C (Filtering). The main steps involved in Task 7a are as follows:

1. **Parameter Iteration**: We iterate over different values of window length and spacing. The `windowrange` list contains the window lengths, and the `spacingrange` list contains the spacings.

2. **Data Loading**: We load the dataset of network traffic into `XX` and `yy`. The dataset is located in the folder `/content/Datasets`, and the filenames are structured based on the spacings and window lengths.

3. **Data Normalization and Dataset Split**: The dataset is normalized using the StandardScaler, and then it is split into training and testing sets. We use 80% of the data for training and 20% for testing.

4. **Model Training**: We train two types of models: Logistic Regression and Deep Neural Network (DNN). The best hyperparameters obtained from tasks 4b) and 4d) are used for training the models.

5. **Predict and Performance Evaluation**: The trained models are used to predict the failure classes on the test set. We evaluate the performance of the models using accuracy, precision, recall, and F1-score. The evaluation results are saved in separate files for each combination of spacing and window length.

The goal of Task 7a is to accurately identify the causes of failure in network communication and assess the performance of the models in this context.


## Homework 2- Traffic Prediction 

(Traffic Prediction and Cost Evaluation)

This folder contains the code for traffic prediction and cost evaluation using ground-truth and predicted traffic traces with Artificial Neural Network (ANN) and Long Short-Term Memory (LSTM) models.

### Task 8a - Traffic Scaling and Visualization

In Task 8a, we calculate the minimum and maximum traffic values of the dataframe created in Task 2b. The ground-truth, ANN-predicted, and LSTM-predicted traffic traces (test set) are then scaled to have a maximum traffic of 1 Gbit/s. The traffic traces are first normalized to the range [0, 1], and then they are converted to CDR units. Finally, we plot the three traffic traces in a single plot.

### Task 8b - Cost Evaluation

In Task 8b, we define the function `evaluate_cost()` to evaluate the cost of over/under-provisioning for the ANN and LSTM predictions. The cost is calculated based on two scalar parameters, `alpha` and `beta`, which represent the weight for over- and under-provisioning costs, respectively.

The function assumes that an operator allocates interfaces with 100 Mbit/s granularity, and it calculates the number of interfaces required for both the ground-truth and predicted traffic traces. The cost of over/under-provisioning is then calculated by weighting the difference between the ground-truth and predicted traffic with the parameters `alpha` and `beta`. Over-provisioning cost can be associated with unnecessary power consumption, while under-provisioning cost can be due to blocked traffic.

### Task 8c - 3D Cost Visualization

In Task 8c, we visualize the costs for different values of `alpha` and `beta` in a 3D plot. The costs are calculated for various combinations of `alpha` (ranging from 2 to 4) and `beta` (ranging from 10 to 30 with a step of 10). The plot shows the cost for both ANN and LSTM predictions, and the bars represent the cost values for each combination of `alpha` and `beta`.

The cost evaluation helps in understanding the impact of different resource allocation policies on network performance and can be used to optimize the prediction models based on specific cost constraints.

Please Note: The description for Task 8c is not provided in the notebook cells, so additional details are required to include it in this repository's README.

## Homework 3- QoS Estimation
(Uncertain Features and Performance Evaluation)

This repository contains code for traffic prediction using Neural Networks (NN) with uncertain features. In Task 7a, we define the function `extract_UNCERTAIN_features()` to generate span length features with a random error chosen from a normal distribution with a mean of 0 and a standard deviation (*sigma*) passed as input. The function takes in the matrix of spans for each lightpath (*span_matrix*) and the matrix of interferer information (*interferer_matrix*) and returns a numpy array *X_matrix* containing the following features for each lightpath:

1. Number of fiber spans along the path
2. Total lightpath length
3. Longest fiber span length (between two amplifiers)
4. Maximum number of interferers across all the links traversed
5. Frequency distance from the closest interferer across all the links traversed (if *interferer_flag* is set to *True*)

In Task 7b, we consider the NN algorithm only and redo the training and performance evaluation using a new dataset with uncertain features. For each error standard deviation, the following steps are performed:

1. Generate a new features matrix with uncertain features using the function from Task 7a.
2. Scale and split the dataset.
3. Train a new NN model on the new dataset.
4. Predict and evaluate the performance of the NN model.

The error standard deviation is introduced in the span length with values of 5%, 10%, and 15% of the maximum span length across all lightpaths. The dataset is generated from the file `SNR_dataset_0.25dB_per_km.txt`, and the new features matrix is used to train the NN model with varying error levels.

The performance evaluation provides insights into how the uncertainty in the features affects the accuracy of the NN model's predictions and helps in understanding the robustness of the model under varying conditions.
