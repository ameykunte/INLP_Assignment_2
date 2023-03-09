# Step 1: Load the Data
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Embedding, SimpleRNN, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import os


def load_data(path):
    sentences = []
    tags = []
    words = set()
    tag_set = set()

    with open(path, 'r') as f:
        lines = f.readlines()
        sentence = []
        sentence_tags = []
        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                tags.append(sentence_tags)
                sentence = []
                sentence_tags = []
             
            else:
             values = line.strip().split('\t')
             if len(values) == 3:
               word, root, tag = values
               sentence.append(root)
               sentence_tags.append(tag)
               words.add(root)
               tag_set.add(tag)

    return sentences, tags, words, tag_set


train_sentences, train_tags, train_words, train_tag_set = load_data(
    'safe_train.txt')
test_sentences, test_tags, test_words, test_tag_set = load_data(
    'safe_test.txt')
dev_sentences, dev_tags, dev_words, dev_tag_set = load_data('safe_dev.txt')

# Step 2: Preprocess the Data

# Tokenize the words
tokenizer = Tokenizer(num_words=None)
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)
dev_sequences = tokenizer.texts_to_sequences(dev_sentences)

# Pad the sequences
maxlen = max(len(seq) for seq in train_sequences)
train_sequences = pad_sequences(train_sequences, maxlen=maxlen)
test_sequences = pad_sequences(test_sequences, maxlen=maxlen)
dev_sequences = pad_sequences(dev_sequences, maxlen=maxlen)

# Get the number of words and POS tags
num_words = len(tokenizer.word_index) + 1
num_classes = len(train_tag_set)

# Step 3: Prepare the Data

# Split the training data into training and validation sets
train_sequences, val_sequences, train_tags, val_tags = train_test_split(
    train_sequences, train_tags, test_size=0.2, random_state=42)

# Step 4: Model Architecture

embedding_size = 32
rnn_units = 64

model = Sequential()
model.add(Embedding(input_dim=num_words,
          output_dim=embedding_size, input_length=maxlen))
model.add(SimpleRNN(units=rnn_units, return_sequences=False))
model.add(Dense(units=num_classes, activation='softmax'))

model.summary()

# Step 5: Model Training

batch_size = 128
epochs = 500

# Convert the tags to one-hot encoding
train_tags_onehot = to_categorical(train_tags, num_classes=num_classes)
val_tags_onehot = to_categorical(val_tags, num_classes=num_classes)
test_tags_onehot = to_categorical(test_tags, num_classes=num_classes)
dev_tags_onehot = to_categorical(dev_tags, num_classes=num_classes)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_sequences, train_tags_onehot, batch_size=batch_size,
          epochs=epochs, validation_data=(val_sequences, val_tags_onehot))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_sequences, test_tags_onehot)
print("Test loss:", loss)
print("Test accuracy:", accuracy)