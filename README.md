# Image Classification with TensorFlow and Tkinter
This project demonstrates a simple image classification application using TensorFlow and Tkinter. The model is built on the MobileNetV2 architecture and trained to recognize images of animals. The application provides a graphical user interface (GUI) for users to upload and classify images.
# Features
* Model Training: Utilizes MobileNetV2 pre-trained on ImageNet, with additional custom layers for animal classification.
* Data Augmentation: Implements various data augmentation techniques to improve model generalization.
* GUI Application: Built with Tkinter, allowing users to open and classify images directly from a user-friendly interface.
# Components

1. **Model Architecture**:
   • MobileNetV2 as the base model with weights pre-trained on ImageNet.
   • Additional layers for global average pooling and dense classification.
   • Compiled with Adam optimizer and categorical crossentropy loss.

2. **Data Preparation**:
   • Uses `ImageDataGenerator` for training and validation data.
   • Performs data augmentation including rotation, shifting, shearing, zooming, and flipping.

3. **Image Classification**:
   • Preprocesses input images to match the model's expected input format.
   • Predicts the class of the input image and maps the predicted index to the class label.

4. **GUI Application**:
   • Built with Tkinter to open and display images.
   • Shows the predicted class of the uploaded image.

# Installation
1. **Clone the repository**:
   ```bash
    git clone [repository-url](https://github.com/username/repository.git)
3. **Install the required packages**: Run the following command:
   ```bash
   pip install tensorflow pillow numpy

# Notes
* Update the paths for train_dir and validation_dir to point to your dataset.
* Ensure the dataset contains subdirectories for each class with images.
# Example Output

![image](https://github.com/user-attachments/assets/c5da1f1d-902a-4bdb-a77e-22547df2fcd8)
![image](https://github.com/user-attachments/assets/0e83fab2-7c04-489a-b9e8-8adfc1d19d71)
![image](https://github.com/user-attachments/assets/7ea9bcc3-9684-414f-9831-a4c04d7633e2)
![image](https://github.com/user-attachments/assets/a2abed90-239e-4726-b21b-235a0bf0c005)
![image](https://github.com/user-attachments/assets/47339937-3680-4ea7-bf17-76e08704699d)
![image](https://github.com/user-attachments/assets/d72b3083-1c14-41ee-aa9c-91a4139c8e7f)

