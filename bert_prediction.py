import numpy as np

import global_variables
from models import BERTEmotionModelReloaded

# Get the model.
model = BERTEmotionModelReloaded(
    global_variables.DIR_FINAL_MODEL_BERT)

try:
    text = input('Enter a text: ')
except:
    text = 'what a nice weather'

results = model.predict(text)

input_text = results['text']
emotion = results['emotion']
confidence_values = np.fromiter(
    results['coefficients'].values(), dtype=float)
confidence = np.max(np.ravel(confidence_values))

print(f'Text : {input_text}')
print(f'Emotion : {emotion}')
print(f'Confidence : {confidence}')