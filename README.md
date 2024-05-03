# 2024-AI-Challenge---The-A-game

Overview of product - ArtSense

This project aims to classify artworks into two categories: AI-generated art and human-created art. It utilizes Convolutional Neural Networks (CNNs) for image classification, focusing on distinguishing between artworks created by artificial intelligence algorithms and those created by human artists.

# Table of Contents

**1)Introduction**

**2)Dataset**

**3)Model Architecture**

**4)Usage**

**5)Results**

**6)Future Improvements**

**Introduction**

Artificial intelligence has made significant strides in generating art that can often be indistinguishable from human-created artworks. This project explores the use of CNNs, a type of deep learning model well-suited for image classification tasks, to differentiate between AI-generated and human-created art pieces. It also uses code to generate the training/testing data for ai generated labeled data.

**Dataset**

The dataset used for training and testing the CNN model consists of two classes:

AI-generated art images

Human-created art images

The dataset is sourced from "AI and Human generated art" by kausthab kannan on kaggle and contains a diverse collection of artworks in various styles and genres.

**Model Architecture**

The CNN model architecture used for this project comprises several convolutional layers followed by pooling layers and fully connected layers. The model is trained using the dataset to learn the features that distinguish between AI-generated and human-created art. 
Stable Diffusion is a deep learning model used for converting text to images. It is recommended to use the model with high powered NVIDIA GPU compute for faster image generation.

**Usage**

To use this project, follow these steps:

-Clone the repository to your local machine.

-Install the necessary dependencies listed in requirements.txt.

-Enter the sample dataset image name in line 28 of the test.py file beside imageFile variable (Ai generated sample 2 is added by default).

-Run the test.py script to test the CNN model with sample data.

In the generate.py file enter your prompt to generate an image on line 8 beside propmpt variable.

-Evaluate the model using the generate.py file to generate test dataset and assess its performance (takes up lots of gpu compute- NVIDIA compute recommended).

-Use the trained/tested model for classifying new art images into AI-generated or human-created categories.


Example commands:

bash

Copy code

git clone https://github.com/ashshahzad/2024-AI-Challenge---The-A-game.git

cd 2024-AI---ChallengeThe-A-game

pip install -r requirements.txt

**Organize sample dataset, test model, and evaluate**

python test.py

python generate.py


**Results**

![image](https://github.com/ashshahzad/2024-AI-Challenge---The-A-game/assets/153488884/6e8e86a1-d588-455a-b60e-23aa5db788af)

The trained CNN model achieves 89% precision, 89% recall and 95% accuracy on the test dataset, demonstrating its effectiveness in classifying AI and human art.

Sample classification results:

AI-generated art: "AI generated S1.jpg"

Human-created art: "Human generated S1.jpg"

**Future Improvements**

Future enhancements to this project may include:

Fine-tuning the model architecture for better performance.

Exploring transfer learning techniques using pre-trained CNN models.

Enhancing the dataset with more diverse artworks and labels.

Implementing a user interface for interactive art classification.
