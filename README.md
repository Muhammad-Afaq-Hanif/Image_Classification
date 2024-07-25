# Image Classification with TensorFlow and Tkinter
This project demonstrates a simple image classification application using TensorFlow and Tkinter. The model is built on the MobileNetV2 architecture and trained to recognize images of animals. The application provides a graphical user interface (GUI) for users to upload and classify images.
# Features
* Model Training: Utilizes MobileNetV2 pre-trained on ImageNet, with additional custom layers for animal classification.
* Data Augmentation: Implements various data augmentation techniques to improve model generalization.
* GUI Application: Built with Tkinter, allowing users to open and classify images directly from a user-friendly interface.
# Components
1. **Model Architecture**:
   - MobileNetV2 as the base model with weights pre-trained on ImageNet.
   - Additional layers for global average pooling and dense classification.
   - Compiled with Adam optimizer and categorical crossentropy loss.

2. **Data Preparation**:
   - Uses `ImageDataGenerator` for training and validation data.
   - Performs data augmentation including rotation, shifting, shearing, zooming, and flipping.

3. **Image Classification**:
   - Preprocesses input images to match the model's expected input format.
   - Predicts the class of the input image and maps the predicted index to the class label.

4. **GUI Application**:
   - Built with Tkinter to open and display images.
   - Shows the predicted class of the uploaded image.
# Installation
1. **Clone the repository**:
git clone <repository-url>
2. **Install the required packages**:
pip install tensorflow pillow numpy
# Notes
* Update the paths for train_dir and validation_dir to point to your dataset.
* Ensure the dataset contains subdirectories for each class with images.
# Example Output


![image](https://github.com/user-attachments/assets/81ae8bbb-7163-48a3-a02f-2d767e1832de)
![image](https://github.com/user-attachments/assets/6449e2a7-e80c-484e-b075-30cd8e75836d)
![image](https://github.com/user-attachments/assets/7b07b8aa-0ac8-43c2-8149-96b96401e252)
![image](https://github.com/user-attachments/assets/74429d59-5298-44b5-a844-5b191d25bdb1)
![image](https://github.com/user-attachments/assets/56ed82e4-af0c-45e7-a218-369a1d3afb06)

