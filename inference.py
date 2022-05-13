from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import json
import os

if __name__=='__main__':
    # Load the model
    model = load_model(os.path.join('pretrained', 'keras_model.h5'))

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    image_list = [img for img in os.listdir('images') if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg')]

    predictions = {}
    for image_name in image_list:
        # Replace this with the path to your image
        image = Image.open(os.path.join('images', image_name))
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        
        predictions[image_name] = prediction.tolist()

    if not os.path.exists('result'):
        os.makedirs('result')
    
    with open(os.path.join('result', 'predictions.json'), 'w') as f:
        json.dump(predictions, f)