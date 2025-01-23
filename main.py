#after training model
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model_path = 'path/for/model/model.h5'
model = load_model(model_path)

img_path = 'pth/for/img'

img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype('float32') / 255.0  # Assuming the model expects values in the range [0, 1]

# Add batch dimension to the image (from (256, 256, 3) to (1, 256, 256, 3))
img = np.expand_dims(img, axis=0)

# Predict using the model
predictions = model.predict(img)
plt.imshow(predictions[0])  # Assuming the model outputs an image of the same shape
plt.show()
