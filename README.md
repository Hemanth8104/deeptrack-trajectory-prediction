# deeptrack-trajectory-predictionDeepTrack: Lightweight Deep Learning for Vehicle Trajectory Prediction in Highways

This project implements a vehicle trajectory prediction model inspired by the DeepTrack model from the paper 
"DeepTrack: Lightweight Deep Learning for Vehicle Trajectory Prediction in Highways".

--------------------------------------------------------------------------------
Table of Contents
--------------------------------------------------------------------------------
1. Overview
2. Features
3. Installation
4. Usage
    4.1 Dataset Preparation
    4.2 Training
    4.3 Evaluation
5. Results
6. References

--------------------------------------------------------------------------------
1. Overview
--------------------------------------------------------------------------------
The DeepTrack model uses a modified Temporal Convolutional Network (TCN) for efficient and real-time vehicle 
trajectory prediction. This repository provides code for data loading, model definition, training, and evaluation.

--------------------------------------------------------------------------------
2. Features
--------------------------------------------------------------------------------
- Custom Dataset Loader: Loads the NGSIM dataset or generates synthetic data if needed.
- Temporal Convolutional Network (TCN) Model: TCN architecture with depthwise convolutions to improve efficiency.
- Evaluation Metrics: Includes Mean Squared Error (MSE) and accuracy checking based on a distance threshold.

--------------------------------------------------------------------------------
3. Installation
--------------------------------------------------------------------------------
1. Clone the repository:
   git clone https://github.com/your-username/deeptrack-trajectory-prediction.git
   cd deeptrack-trajectory-prediction

2. Install the required dependencies:
   pip install -r requirements.txt

--------------------------------------------------------------------------------
4. Usage
--------------------------------------------------------------------------------
4.1 Dataset Preparation
    - To use the NGSIM dataset, download the data and specify the data_path in the code.
    - Alternatively, synthetic data can be generated automatically if no path is provided.

4.2 Training
    - To train the DeepTrack model, run the following command:
      python deeptrack_training.py --data_path /path/to/ngsim.csv --epochs 20

4.3 Evaluation
    - To evaluate the model on a test set, use the following command:
      python deeptrack_training.py --data_path /path/to/ngsim.csv --eval

--------------------------------------------------------------------------------
5. Results
--------------------------------------------------------------------------------
After training, you can visualize training and test MSE loss values over epochs.

--------------------------------------------------------------------------------
6. References
--------------------------------------------------------------------------------
Katariya, V., Baharani, M., Morris, N., Shoghli, O., & Tabkhi, H. (2022). 
"DeepTrack: Lightweight Deep Learning for Vehicle Trajectory Prediction in Highways."
IEEE Transactions on Intelligent Transportation Systems, 23(10), 18927-18935.

--------------------------------------------------------------------------------
File Descriptions
--------------------------------------------------------------------------------
- README.txt                : Overview, setup, and usage instructions
- requirements.txt          : Lists Python dependencies for this project
- deeptrack_training.py     : Main script for loading data, training, and evaluating the model
