 # Cmpe591 Deep Learning for Robotics Project 1: 
Drive link for network weights as they are too large to include in the git repo: https://drive.google.com/drive/folders/1z2JmUylu1evV_xS4hHgZiqQu_Ooa80Hx

## Overview
This project consists of three tasks aimed at training neural networks to predict the effect of a robot's actions on an object placed on a table. The dataset includes:
- **Input:** An image of the initial state and a discrete action (1, 2, 3, or 4).
- **Output:** The final position of the object.

## Networks
Every network has a decicated .py file. Inside every file, there is a "train" and "test" functions. Usage of these functions with appropriate arguments are given in the main block in each file.

1. **MLP (Multi-Layer Perceptron) for Position Prediction**  
   - Predicts the final object position `(x, y)` given the initial image and action.  
   - Uses a **fully connected network (MLP)** for regression.
 
Loss graph:

![DNN](https://github.com/user-attachments/assets/fa221308-e556-4008-b668-e31799ace0c3)

2. **CNN (Convolutional Neural Network) for Position Prediction**  
   - Predicts the final object position `(x, y)`, similar to Task 1.  
   - Uses a **CNN encoder** to extract features before passing them to an MLP.
  
Loss graph:

![cnn](https://github.com/user-attachments/assets/14b7ce85-ddf5-441b-aa98-b444728f45ec)

3. **CNN for Image Reconstruction**  
   - Predicts the **final state image** given the initial image and action.  
   - Uses a **CNN encoder-decoder architecture** for image-to-image translation.
  
Loss graph:

![reconstructor_loss_graph](https://github.com/user-attachments/assets/b64ee116-d8e9-4282-8713-0c27ef46e964)


Target Output Example:

![output_target](https://github.com/user-attachments/assets/3146e1ea-acd8-40a5-85c5-f594f38c3069)

Output Example:

![myplot](https://github.com/user-attachments/assets/fb354551-b3c4-44ef-ad7c-28472eda1138)




## Repository Structure

    ├── data/
    │   ├── actions/                         # Folder containing action data
    │   ├── images/                          # Folder containing image data
    │   ├── positions/                       # Folder containing position data
    ├── best_cnn.pth                         # Pretrained model for CNN
    ├── best_DNN.pth                         # Pretrained model for DNN
    ├── best_cnn_reconstruction.pth          # Pretrained model for image reconstructing CNN
    ├── CNN.py                               # CNN model for position prediction
    ├── CNN_Reconstruction.py                # CNN model for image reconstruction
    ├── DNN.py                               # DNN model for position prediction
    ├── homework1.py                         # Environment setup, Dataset class and data collection
    ├── Hw1Env.py                            # Simulation environment base file

## Installation & Dependencies
I used a conda virtual environment for the project, exact environment details are given in the environment.yml file. 
For installing the virtual environment:

    conda env create -f environment.yml

Then activate it using:

    conda activate my_env  # Replace `my_env` with the desired environment name
