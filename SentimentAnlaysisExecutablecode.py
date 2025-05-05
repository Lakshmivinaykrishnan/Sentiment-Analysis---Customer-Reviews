#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of User Reviews: Uncovering Customer Opinions
# ### Can neural networks and NLP techniques accurately classify customer sentiment to benefit organizations? 
# #### The goal of sentiment or opinion analysis to get customer emotions from the reviews, whether it is positive or negative

# In[67]:


# Libraries
import sklearn  # model evaluation and splitting datasets
import pandas as pd  # handling and manipulating structured data (dataframes)

# Natural language processing (text preprocessing)
import re  # library for pattern matching and text preprocessing
import nltk  # Natural Language Toolkit for processing and analyzing textual data
from nltk.corpus import stopwords  # list of common stop words for various languages
from langdetect import detect, DetectorFactory  # Detects the language of a given text

# Text analysis and visualization
from wordcloud import WordCloud  

# Visualization libraries
import matplotlib.pyplot as plt 
import seaborn as sns  

# Keras utilities for neural network construction and preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Prepares text sequences to have equal lengths
import tensorflow as tf  # TensorFlow library for building and training machine learning models
from tensorflow.keras.layers import (  # Layers used to define the neural network architecture
    Embedding,  # Converts words into dense vectors (embeddings)
    Dropout,  # Regularization layer to prevent overfitting
    Bidirectional,  # Wraps an RNN layer to process input in both forward and backward directions
    LSTM,  # Long Short-Term Memory layer for sequential data processing
    Dense  # Fully connected layer
)
from tensorflow.keras.models import Sequential  #Define the model
from tensorflow.keras.callbacks import EarlyStopping  # Stops training early if validation performance stops improving

# Tokenizer for text preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer  # Converts text into numerical token sequences

# Scikit-learn utilities for model evaluation
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from sklearn.metrics import ConfusionMatrixDisplay
# General-purpose utilities
import sys  # Provides access to system-specific parameters and functions
import warnings  # Used to control warning messages


# In[68]:


# install nltk
# NLTK: Natural Language tool kit
#!pip install nltk


# In[69]:


from platform import python_version
print("\n python version is ",python_version())


# ### Load Data-Converting Text file into file

# In[70]:


amazon=pd.read_csv('path/amazon_cells_labelled.txt',sep='\t',header=None, names=['review', 'sentiment'])
# Display the DataFrame info to verify
print(amazon.info())
print(amazon.head())


# In[71]:


yelp=pd.read_csv('path/yelp_labelled.txt',sep='\t',header=None, names=['review', 'sentiment'])
# Display the DataFrame info to verify
print(yelp.info())
print(yelp.head())


# In[72]:


# Read the file line by line and process
with open('path/imdb_labelled.txt', 'r') as file:
    # Split by tab and filter out malformed lines
    lines = [line.strip().split("\t") for line in file if len(line.strip().split("\t")) == 2 and line.strip().split("\t")[1] != '']

# Convert the processed data into a DataFrame
imdb = pd.DataFrame(lines, columns=['review', 'sentiment'])

# Display the DataFrame info to verify
print(imdb.info())
print(imdb.head())


# In[73]:


print(amazon.shape)
print(yelp.shape)
print(imdb.shape)


# In[74]:


# Convert all Sentiment values to integers
amazon['sentiment'] = amazon['sentiment'].astype(int)
yelp['sentiment'] = yelp['sentiment'].astype(int)
imdb['sentiment'] = imdb['sentiment'].astype(int)



# In[75]:


# Concatenate the datasets
review_df=pd.concat((amazon,yelp,imdb),ignore_index=True)
print(review_df.info())


# In[76]:


review_df.shape


# In[77]:


#check null values
review_df.isna().sum()


# In[78]:


review_df.tail(5)


# ### Visualizing statements in the review

# In[79]:


print(review_df.columns)


# In[80]:


# Check unique values in each dataset
print("Amazon Sentiment unique values:", amazon['sentiment'].unique())
print("Yelp Sentiment unique values:", yelp['sentiment'].unique())
print("IMDb Sentiment unique values:", imdb['sentiment'].unique())

# Check unique values in the concatenated dataframe
print("Review DF Sentiment unique values:", review_df['sentiment'].unique())


# In[81]:


# Plot the Sentiment column using countplot
plt.figure(figsize=(8, 6))
sns.countplot(data=review_df, x='sentiment', palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('sentiment')
plt.ylabel('count')
plt.show()


# ## Text Preprocessing

# In[82]:


#print(review_df.duplicated().sum())


# In[83]:


#review_df = review_df.drop_duplicates()


# In[84]:


#Initial list of words/characters in reviews
reviews = review_df['review']
list_of_chars = []

for comment in reviews:
    for character in comment:  # Iterate over characters in each review
        if character not in list_of_chars:
            list_of_chars.append(character)

print(list_of_chars)


# In[85]:


review_df.head(5)


# In[86]:


# Extract unique words from reviews
reviews = review_df['review']
list_of_words = []

for comment in reviews:
    words = comment.split()  # Split the comment into words
    for word in words:
        if word not in list_of_words:
            list_of_words.append(word)

#print(list_of_words)

for i in range(0, 20):
    print(list_of_words[i])


# In[87]:


import nltk
nltk.download('punkt')  #Downloads models and datasets for tokenization
nltk.download('wordnet')   #Downloads the WordNet corpus for lemmatization 


# ### N-Gram structures
# #### Unigrams are individual words in the text, Bigrams are pairs of consecutive words, and Trigrams are sequences of three consecutive words

# In[88]:


from sklearn.feature_extraction.text import CountVectorizer


# Define a function to extract n-grams
def extract_ngrams(review_df, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    ngrams_matrix = vectorizer.fit_transform(review_df['review'])
    
    # Get n-grams as feature names
    ngrams = vectorizer.get_feature_names_out()
    
    # Sum the occurrences of each n-gram in the dataset
    ngram_freq = ngrams_matrix.sum(axis=0).A1
    ngram_freq_df = pd.DataFrame(zip(ngrams, ngram_freq), columns=['N-Gram', 'Frequency'])
    
    # Sort the n-grams by frequency
    ngram_freq_df = ngram_freq_df.sort_values(by='Frequency', ascending=False)
    
    return ngram_freq_df

# Example usage: Extracting unigrams, bigrams, and trigrams
unigrams = extract_ngrams(review_df, ngram_range=(1, 1))
bigrams = extract_ngrams(review_df, ngram_range=(2, 2))
trigrams = extract_ngrams(review_df, ngram_range=(3, 3))

# Display the top 10 unigrams, bigrams, and trigrams
print("Top Unigrams:")
print(unigrams.head(10))

print("\nTop Bigrams:")
print(bigrams.head(10))

print("\nTop Trigrams:")
print(trigrams.head(10))


# ### Foreign language presence

# In[89]:


# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Function to detect language
def detect_language(text):
    try:
        # Detect language
        return detect(text)
    except:
        # Handle cases where detection fails
        return None

# Add a 'language' column to the DataFrame
review_df['language'] = review_df['review'].apply(detect_language)

# Display unique languages present in the data
unique_languages = review_df['language'].unique()
print("Unique languages in the dataset:", unique_languages)

# Count occurrences of each language
language_counts = review_df['language'].value_counts()
print("Language counts:\n", language_counts)

# Filter reviews that are in English
en_reviews = review_df[review_df['language'] == 'en'].reset_index(drop=True)

# Drop the 'language' column 
en_reviews = en_reviews.drop(columns=['language'])

print(en_reviews)


# ### Stop words
# #### words in a language that typically carry little meaningful information by themselves 

# ### Removes punctuation, converting to lower case, tokenization, removes stopwords,lemmatization

# In[90]:


#from nltk.corpus import stopwords
# download stopwords
nltk.download("stopwords")
# Get the list of English stopwords
stopwords_list = stopwords.words('english')
custom_stopwords = ["n't", "'s", "'re", "'ve", "'m", "'ll", "'d"]  # Add common contractions
stopwords_list.extend(custom_stopwords)

print(stopwords_list)


# In[91]:


#pip install langdetect


# In[92]:


#print(english_reviews.head(5))
print(review_df.shape)
print(en_reviews.shape)


# #### Unusual characters found

# In[93]:


# Initialize an empty list to store unusual characters
unusual_characters = []

# Loop through each review in the 'review' column
for review in review_df['review']:
    # Loop through each character in the review
    for character in review:
        # Check if the character is not alphanumeric (a-z, A-Z, 0-9) or a space
        if not re.match(r"[a-z\s]", character) and character not in unusual_characters:
            # Append the unusual character to the list
            unusual_characters.append(character)

# Print the list of unusual characters found in the reviews
print("Unusual Characters Found:", unusual_characters)


# In[94]:


# Initialize an empty list to store processed descriptions
description_list = []

# Regular expression, removes punctuation and special characters
for description in en_reviews['review']:
    # Remove non-alphabetic characters 
    description = re.sub("[^a-zA-Z]", " ", description)
    description= re.sub(r'#\w+', ' ', description) # hashtags
    description= re.sub(r'http\S+', ' ', description) # links
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    description = emoji_pattern.sub(r'', description)  # Remove emojis
   
    # Convert to lowercase
    description = description.lower()

   
    # Perform tokenization, Splits the text into individual words 
    description = nltk.word_tokenize(description)
    
    # Remove stopwords
    description= [word for word in description if word not in stopwords_list]

    # Perform lemmatization
    lemma = nltk.WordNetLemmatizer()      # tool that reduces words to their root form (lemma), e.g: running->run,movies->movie
    description = [lemma.lemmatize(word) for word in description]

    

    # Append the processed description to the list,build a final cleaned tokenized dataset
    description_list.append(description)
    
print(description_list[:10])


# Calculate the length of each word in the processed descriptions
word_lengths = [len(word) for description in description_list for word in description]

# Plot a histogram of the word lengths
plt.figure(figsize=(10, 6))
plt.hist(word_lengths, bins=30, edgecolor='black')
plt.title('Histogram of Word Lengths in Cleaned Reviews')
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.show()


# In[95]:


max(word_lengths)


# #### most frequent word lengths are between 4 to 5 characters
# #### X-Axis shows the length of the words, ranging from 1 to 16 characters
# #### Y-Axis  represents how many words of a particular length are present in the dataset

# In[96]:


# Assign the cleaned descriptions to a new column in the dataframe
en_reviews['cleaned_review'] = description_list

# Print a sample of the dataframe to verify
print(en_reviews.head())


# ### Word cloud for positive and negative reviews
# #### Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text.

# In[97]:


#!pip install wordcloud


# In[98]:


# Separate positive and negative reviews
positive_reviews = en_reviews[en_reviews['sentiment'] == 1]['cleaned_review']
negative_reviews = en_reviews[en_reviews['sentiment'] == 0]['cleaned_review']

# Combine the tokenized words into single strings for word cloud generation
positive_text = " ".join([" ".join(review) for review in positive_reviews])
negative_text = " ".join([" ".join(review) for review in negative_reviews])

# Generate word clouds
positive_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_text)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(negative_text)

# Plot the word clouds
plt.figure(figsize=(14, 7))

# Positive review word cloud
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title("Word Cloud for Positive Reviews", fontsize=16)
plt.axis('off')

# Negative review word cloud
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title("Word Cloud for Negative Reviews", fontsize=16)
plt.axis('off')

plt.tight_layout()
plt.show()


# #### Vocabulary size

# In[99]:


#identify vocabulary size

# Initialize the tokenizer
tokenizer=Tokenizer()

# Fit the tokenizer on the 'cleaned_review' column of the dataframe
tokenizer.fit_on_texts(en_reviews['cleaned_review'])
vocab_size=len(tokenizer.word_index)+1
print("Vocabulary_size:",vocab_size)  # Add 1 for padding token


# #### Text statistics

# In[100]:


import numpy as np

# Loop through the reviews to calculate their lengths
review_length = [len(i) for i in en_reviews['cleaned_review']]

# Calculate the statistics
review_min = np.min(review_length)
review_median = np.median(review_length)
review_max = np.max(review_length)#Text statistics


# Display the results
print(f"Minimum review length: {review_min}")
print(f"Median review length: {review_median}")
print(f"Maximum review length: {review_max}")


# Find the index of the longest and shortest review
min_length_idx = np.argmin(review_length)
max_length_idx = np.argmax(review_length)

# Get the longest and shortest reviews
shortest_review = en_reviews['cleaned_review'].iloc[min_length_idx]
longest_review = en_reviews['cleaned_review'].iloc[max_length_idx]


# Display the shortest and longest reviews with their lengths
print(f"\nShortest Review (Length: {review_min} characters):")
print(shortest_review)
print(f"\nLongest Review (Length: {review_max} characters):")
print(longest_review)


# ###  reviews are being tokenized into lists of words, minimum and maximum lengths are number of words

# In[101]:


#same as previous vocab_size code to find the total vocabulary size

list_of_words=[]
for i in en_reviews['cleaned_review']:
    for j in i:
        list_of_words.append(j)

#list_of_words
# Obtain the total number of unique words
total_words = len(list(set(list_of_words)))
total_words


# ### Split the data into Train and Test

# In[102]:


#from sklearn.model_selection import train_test_split

# Ensure sentiment column contains only integers
en_reviews['sentiment'] = en_reviews['sentiment'].astype(int)

# Convert description_list into NumPy array with string format for compatibility
x = np.array([" ".join(description) for description in description_list])  # Join tokenized words into strings
y = en_reviews['sentiment'].values  # Target variable

# Split the data into train and test sets
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=15, stratify=y)
x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
    x, y, range(len(en_reviews)), test_size=0.20, random_state=15, stratify=y
)

# Convert y_train and y_test into pandas Series
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# View the training and testing sample sizes
print("Training size: ", x_train.shape)
print("Testing size: ", x_test.shape)


# In[103]:


x_train[:10]


# In[104]:


x_test[:10]


# ## Apply  padding 
# #### Padding is the process of ensuring that all input sequences in a dataset have the same length by adding zeros (or another specified value) to shorter sequences. This is necessary because many machine learning models, require inputs of uniform length.

# In[105]:


#from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure the tokenizer is fitted on the entire dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)  # Fit tokenizer on all text data (x)

# Calculate the maximum sequence length from the training data only
max_length = max([len(text.split()) for text in x_train])  # Calculate max length from training set to avoid data leakage

# Print the derived max_length to verify
print(f"Max sequence length from training set: {max_length}")

# Define padding parameters
padding_type = 'post'  #adds padding at the end of the sequence 
trunc_type = 'post'  # truncating at the end if sequences exceeds max limit

# Apply padding to training data
sequences_train = tokenizer.texts_to_sequences(x_train)  # Convert training text to sequences
padded_train = pad_sequences(sequences_train, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Apply padding to testing data
sequences_test = tokenizer.texts_to_sequences(x_test)  # Convert testing text to sequences
padded_test = pad_sequences(sequences_test, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Display the shapes of padded arrays
print("Padded training data shape: ", padded_train.shape)
print("Padded testing data shape: ", padded_test.shape)


# In[106]:


# Sample text data
sample = ["sample word for tokenization", "I love NLP"]

# Initialize the tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on the text data (sample)
tokenizer.fit_on_texts(sample)

# Display the word index (word to integer mapping)
print("Word Index:", tokenizer.word_index)

# Convert the texts into sequences of integers (tokenized)
sequences = tokenizer.texts_to_sequences(sample)

# Display the tokenized sequences
print("Tokenized Sequences:", sequences)


# In[107]:


#import sys

# Set the print options to display the full array 
np.set_printoptions(threshold=sys.maxsize)

# Print the padded sequence (adjust index )
print(padded_train[-1])



# In[108]:


print(len(padded_train[1])) 


# In[109]:


print("The encoding for review \n", x_train[1:2],"\n is: ", padded_train[1])


# In[110]:


print("The encoding for review\n", x_test[1:2],"\n is: ", padded_test[1])


# In[111]:


from tensorflow.keras.utils import to_categorical

# Convert the data to categorical 2D representation, will be useful if RNN model implements
#categorical_crossentropy expects one-hot encoded labels, while sparse_categorical_crossentropy works with integer labels.
# Not required while binary_cross_entropy
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)
print(y_train_cat.shape)
print(y_test_cat.shape)


# In[112]:


for i in range(0,10): print(y_train_cat[i])


# #### convert padded data to  numpy array to be used in model

# In[113]:


#convert padded data to  numpy array to be used in model
training_padded=np.array(padded_train)
training_label=np.array(y_train)
test_padded=np.array(padded_test)
test_label=np.array(y_test)


# In[114]:


#export the data to csv
pd.DataFrame(training_padded).to_csv("path/training_padded.csv")
pd.DataFrame(training_label).to_csv("path/training_label.csv")
pd.DataFrame(test_padded).to_csv("path/test_padded.csv")
pd.DataFrame(test_label).to_csv("path/test_label.csv")


# # Building Neural Network

# #### Long Short Term Memory (LSTM) is a type of RNN designed to capture long-term dependencies
# #### Bidirectional LSTM architecture  processes the sequence in both forward and backward directions.

# #### The embedding layer can handle variable-length input sequences, making it well-suited for tasks like sentiment analysis, where the word order and context matter, atomatically learns the representations of words as dense vectors (embeddings) based on the context they appear in

# In[115]:


######### Initial Model ##########
# Define parameters
activation = 'sigmoid'  # For binary classification
loss = 'binary_crossentropy'  # Loss function for binary classification
optimizer = 'adam'  # Optimizer
num_epochs = 20  # Number of epochs
embedding_dim = 100  # Dimension of embedding vectors
batch_size = 32  # Batch size for training


# Early stopping to prevent overfitting
early_stopping_monitor = EarlyStopping(patience=2)

# Build the RNN-based model
model_lstm = Sequential([
    # Embedding layer to convert tokenized words into dense vectors
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Dropout(0.3),  # Regularization to prevent overfitting
    
    # Bidirectional LSTM layer to capture context from both directions
    Bidirectional(LSTM(units=64, dropout=0.5, recurrent_dropout=0.4)),
           
    # Output layer with sigmoid activation for binary classification
    Dense(1, activation=activation)
])

# Compile the model
model_lstm.compile(
    loss=loss,  
    optimizer=optimizer,  
    metrics=['accuracy']  # Metrics to monitor during training
)

# Train the model
history_lstm = model_lstm.fit(
    training_padded,  # Input training data (tokenized and padded sequences)
    training_label,  # Corresponding labels for training data
    batch_size=batch_size,  # Batch size
    epochs=num_epochs,  # Number of epochs
    validation_split=0.2,  # Use 20% of training data for validation
    callbacks=[early_stopping_monitor],  # Early stopping callback
    verbose=1  # Verbosity level
)

# Display the model summary
model_lstm.summary()

# Evaluate the model on test data
test_loss, test_accuracy = model_lstm.evaluate(test_padded, test_label, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# ### Tuned model

# #### Dimension of embedding vectors increased to 250
# #### Added a 'relu' activation in bidirectional layer, units increased to 128
# #### Additional dense layer has introduced with relu activation and 128 units
# #### Droput layer to prevent overfitting after dense 128 layer
# #### Number of epochs reduced to 10

# In[116]:


#####  Tuned Model ###########
# Define parameters
activation = 'sigmoid'  # For binary classification
loss = 'binary_crossentropy'  # Loss function for binary classification
optimizer = 'adam'  # Optimizer
num_epochs = 10  # Number of epochs reduced to 10
embedding_dim = 250  # Dimension of embedding vectors increased to 250
batch_size = 32 # Batch size for training
early_stopping_monitor = EarlyStopping(patience=2)

# Build the RNN-based model
model_lstm2 = Sequential([
    # Embedding layer 
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    Dropout(0.5),  # Regularization to prevent overfitting
    
    # Bidirectional LSTM layer to capture context from both directions
    Bidirectional(LSTM(units=128, dropout=0.5, recurrent_dropout=0.4,activation='relu')),   
    #Added a 'relu' activation in the bidirectional layer, units increased to 128
    
    Dense(128, activation='relu'),
     Dropout(0.4),
                  
    # Output layer with sigmoid activation for binary classification
    Dense(1, activation=activation)
])

# Compile the model
model_lstm2.compile(
    loss=loss,  
    optimizer=optimizer,  
    metrics=['accuracy']  # Metrics to monitor during training
)

# Train the model
history_lstm2 = model_lstm2.fit(
    training_padded,  
    training_label,  
    batch_size=batch_size,  
    epochs=num_epochs, 
    validation_split=0.2,  
    callbacks=[early_stopping_monitor], 
    verbose=1  
)

# Display the model summary
model_lstm2.summary()

# Evaluate the model on test data
test_loss, test_accuracy = model_lstm2.evaluate(test_padded, test_label, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# In[117]:


# Display the number of layers
num_layers = len(model_lstm2.layers)
print(f"The LSTM model has {num_layers} layers.\n")

# Display the type of each layer
for i, layer in enumerate(model_lstm2.layers):
    print(f"Layer {i+1}: {layer.__class__.__name__}")


# #### Make prediction

# In[118]:


# make prediction
pred = model_lstm2.predict(padded_test)



# In[119]:


print(*pred.flatten())


# In[120]:


predictions = (model_lstm2.predict(padded_test) > 0.5).astype(int)


# In[121]:


# Flatten and print in a single line
print(*predictions.flatten())


# In[122]:


# acuracy score on text data
#from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")



# ### Confusion Matrix

# In[123]:


#from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.show()


# #### The confusion matrix shows a distribution slightly skewed to negative classes with 235 correctly identified classes compared to 222 positive classes. Model also classifies 66 positive and 52 negative classes incorrectly at the time of this execution, values might change slightly on multiple runs. 
# 

# In[124]:


y_test.shape


# ### Test the model

# In[125]:


# Predict probabilities using the padded input data
pred_train = model_lstm2.predict(padded_train)  

train_binary_predictions = (pred_train >= 0.5).astype(int)

pred_test = model_lstm2.predict(padded_test)
test_binary_predictions = (pred_test >= 0.5).astype(int)

# Assign predictions and sentiment labels to the train and test rows in en_reviews
en_reviews.loc[train_indices, 'sentiment_prediction'] = train_binary_predictions
en_reviews.loc[train_indices, 'sentiment_label'] = en_reviews.loc[train_indices, 'sentiment_prediction'].map({0: 'Negative', 1: 'Positive'})

en_reviews.loc[test_indices, 'sentiment_prediction'] = test_binary_predictions
en_reviews.loc[test_indices, 'sentiment_label'] = en_reviews.loc[test_indices, 'sentiment_prediction'].map({0: 'Negative', 1: 'Positive'})

#Verify the result
print(en_reviews[['review','sentiment_prediction','sentiment_label']].head(15))  # Check the first few rows to ensure predictions are assigned correctly


# In[126]:


#Verify the result
print(en_reviews[['review','sentiment_prediction']].head(15))  # Check the first few rows to ensure predictions are assigned correctly


# In[127]:


# Display full string
pd.set_option('display.max_colwidth', None)
print("\n",en_reviews.loc[2213,['review','sentiment_label']])


# ### Incorrect predictions

# In[128]:


# Find the incorrectly predicted reviews
incorrect_predictions = en_reviews[en_reviews['sentiment'] != en_reviews['sentiment_prediction']]

# Display the incorrectly predicted reviews
print("Incorrectly Predicted Reviews:")
print(incorrect_predictions[['review', 'sentiment', 'sentiment_prediction']])


# #### The above results shows the incorrect predictions, which may include mixed sentiments, sarcasm, neutral emotion, domain specific terms and more. Adding more samples with these contexts, Using pre-trained embeddings, data augmentation techniques, fine tune hyperparameters can be considered to enhance the result. Regularly evaluating incorrectly predicted reviews, identifying recurring patterns can help refine the model and address the issues.

# In[129]:


#Export result
pd.DataFrame(en_reviews).to_csv("path/sentiment_model_output.csv")


# In[130]:


# Extract training and validation data from history
history_dict = history_lstm2.history
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(accuracy) + 1)

# Plot accuracy
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Show plots
plt.tight_layout()
plt.show()


# ### Save model

# In[131]:


# Save the text classification model
model_lstm2.save('sentimentAnalysis_bestmodel.keras')


# In[ ]:




