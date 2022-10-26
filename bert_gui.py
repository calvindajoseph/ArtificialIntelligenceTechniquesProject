import numpy as np
import tkinter as tk
from tkinter import ttk

import global_variables
from models import BERTEmotionModelStackedReloaded

class MainApp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        # Configure window.
        self.title('Emotion Detection')
        self.geometry('360x150')
        self.tk.call('tk', 'scaling', 2.0)
        
        self.columnconfigure(3)
        self.rowconfigure(5)
        
        # The text label.
        self.label = ttk.Label(self, text='Text:')
        self.label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Text entry.
        self.entry = ttk.Entry(self, width=50)
        self.entry.grid(row=0, column=1, sticky='nswe', padx=5, pady=5)
        
        # Classify button.
        self.button = ttk.Button(
            self, text='Classify', command=self.classify_text)
        self.button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # Result entry.
        self.result_text = tk.Label(self, text='Results')
        self.result_text.configure(bg='black', fg='white')
        self.result_text.grid(
            row=2, column=0, columnspan=2, padx=5, sticky='nswe')
        
        self.result_emotion = tk.Label(self, text='')
        self.result_emotion.configure(bg='black', fg='white')
        self.result_emotion.grid(
            row=3, column=0, columnspan=2, padx=5, sticky='nswe')
        
        self.result_confidence = tk.Label(self, text='')
        self.result_confidence.configure(bg='black', fg='white')
        self.result_confidence.grid(
            row=4, column=0, columnspan=2, padx=5, sticky='nswe')
        
        # Get the model.
        self.model = BERTEmotionModelStackedReloaded(
            global_variables.DIR_FINAL_MODEL_BERT, 
            global_variables.DIR_FINAL_MODEL_BERT_BINARY)
        
    def classify_text(self):
        text = self.entry.get()
        
        results = self.model.predict(text)
        emotion = results['emotion']
        
        confidence_values = np.fromiter(
            results['coefficients'].values(), dtype=float)
        confidence_value = np.max(np.ravel(confidence_values))
        
        text = f'Text : {text}'
        emotion = f'Emotion : {emotion}'
        confidence = f'Confidence in the main BERT : {confidence_value}'
        
        self.result_text.configure(text=text, anchor='w')
        self.result_emotion.configure(text=emotion, anchor='w')
        self.result_confidence.configure(text=confidence, anchor='w')

if __name__ == '__main__':
    main_app = MainApp()
    main_app.mainloop()