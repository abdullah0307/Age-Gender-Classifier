import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Loading the models
gender_model = load_model("gender_model.h5")
age_model = load_model("Age_model.h5")

# Loading the labels
age_labels = ['(0, 2)', '(15, 20)', '(25, 32)', '(27, 32)', '(38, 42)', '(38, 43)', '(38, 48)', '(4, 6)', '(48, 53)',
              '(60, 100)', '(8, 12)', '(8, 23)']
gender_labels = ['f', 'm']

# Load the face image
image = cv2.imread("My Face.png")

# Applying the preprocessing
# image = image[:1000, 250:750]
resized = cv2.resize(image, (224, 224))
input = np.expand_dims(resized, axis=0)

# Making predictions
g = np.argmax(gender_model.predict(input))
a = np.argmax(age_model.predict(input))

# Displaying it
print(age_labels[a])
print(gender_labels[g])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
