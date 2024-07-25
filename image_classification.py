import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
import numpy as np # type: ignore
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image # type: ignore

num_classes = 90

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers:
    layer.trainable = False

model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_dir = r''# Put path here 
validation_dir = r'' # Put path here 

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

batch_size = 32
image_size = (224, 224)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)


epochs = 1
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)


model.save('my_model.h5')


def preprocess_image(image_path):
    img = load_img(image_path, target_size=image_size)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_label = train_generator.class_indices
    inverse_mapping = dict((v, k) for k, v in class_label.items())
    predicted_class = inverse_mapping[class_index]
    return predicted_class

input_image = r''# Put path here 
predicted_class = predict_image(input_image)


window = tk.Tk()
window.title("Image Recognition")
window.geometry("400x400")

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224))  
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

        predicted_class = predict_image(file_path)
        result_label.configure(text="Predicted class: " + predicted_class)


image_label = tk.Label(window)
image_label.pack(pady=10)

button = tk.Button(window, text="Open Image", command=open_image)
button.pack(pady=10)

result_label = tk.Label(window, text="")
result_label.pack(pady=10)

window.mainloop()
