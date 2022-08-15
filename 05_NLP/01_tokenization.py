import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'IS it sunny today?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

test_data = [
    'Today is a snow day',
    'Will it be rainy tomorrow?'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny day?',
    'I really enjoyed walking in the snow today'
]

padded = pad_sequences(sequences)
print(padded)

padded = pad_sequences(sequences, padding='post')
print(padded)

padded = pad_sequences(sequences, padding='post', maxlen=6)
print(padded)

padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
print(padded)


# HTMLタグの削除
from bs4 import BeautifulSoup
sentence = ""
soup = BeautifulSoup(sentence)
sentence = soup.get_text()

# 句読点の除去
# ストップワードの削除
import string
table = str.maketrans('', '', string.punctuation)
stop_words = ["a", "about", "yours"]
words = sentence.split()
filtered_sentence = ""
for word in words:
    word = word.translate(table)
    if word not in stop_words:
        filtered_sentence = filtered_sentence + word + ' '
sentence.append(filtered_sentence)