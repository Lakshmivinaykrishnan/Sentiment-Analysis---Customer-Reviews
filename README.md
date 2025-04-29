#  Customer Sentiment Analysis
Customer sentiment analysis is an automated process of identifying customer emotions or customer opinions based on user reviews. This analysis is commonly applied to textual data collected from publicly available sources, such as product reviews, survey responses, and user feedback datasets. The goal of this project is to identify sentiment patterns (positive or negative)  and uncover insights into customer satisfaction regarding a product using neural network models and natural language processing techniques. We will use a neural network model and natural language processing (NLP) techniques to analyze customer reviews from three labelled datasets: amazon_cells_labelled, imdb_labelled and yelp_labelled .  By combining the datasets, our aim is to build a model that helps companies better understand customer sentiment and use that information to improve their products and services. Use of multiple datasets will help improve the model's generalizability, ensuring that the findings are robust and applicable across different contexts.

---
###  Datasets Used

1. Amazon Product Reviews
2. Yelp Business Reviews 
3. IMDb Movie Reviews
---
###  Tech Stack

 ##### Python Libraries:  
 - scikit-learn
 - TensorFlow
 - Keras
 - NLTK
 - Pandas
 - Numpy
 - matplotlib
 - langdetect
 - wordcloud
 - re
##### NLP Techniques:
  - Tokenization
  - Lemmatization
  - Stopword Removal
  - Regular expressions
  - Language detection
  - CountVectorizer (Bag-of-Words & N-grams)
  - Padding (preparing input to Neural Networks)
  
---
#### Model Architectures:  
  - Embedding Layer to convert tokens into dense vector representations  
  - LSTM Layer for capturing long-term dependencies in sequential data  
  - Dense Layers for non-linear classification  
  - Dropout layers for regularization  

---
#### Outcome
- Neural networks achieved 80% accuracy on the processed and balanced datasets  
- Padding and word embeddings improved the model's ability to capture context 

