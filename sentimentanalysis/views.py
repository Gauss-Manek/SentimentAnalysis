# views.py
from django.shortcuts import render
from django.http import HttpResponse
import joblib
import os

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Vocabulary size of the tokenizer
vocab_size = 10000

# Maximum length of the padded sequences
max_length = 32

# Parameters for padding and OOV tokens
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# save the model to a file
my_model = joblib.load('model.joblib')


def sentiment_analysis_view(request):
    sentences=[]
    text = request.GET.get('text') 
    sentences.append(str(text))

    # Generate the word index dictionary
    tokenizer.fit_on_texts(sentences)

    # Generate and pad the sequences
    phrase_sequences = tokenizer.texts_to_sequences(sentences)
    phrase_padded = pad_sequences(phrase_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    Modelprediction=my_model.predict(phrase_padded)[0][0]
    
    SarcasmOuPas=''
    if Modelprediction> 0.5:
        SarcasmOuPas='Sarcasm'
    else:
        SarcasmOuPas='Not sarcasm'


    return render(request, 'base.html', {'prediction': SarcasmOuPas})
