# Recommendation Systems: Overview and Methods

Recommendation systems are essential for many modern applications, from streaming platforms like **Netflix** and **YouTube** to e-commerce sites like **Amazon**. They help provide personalized recommendations to users, enhancing user experience and increasing engagement and sales. In this section, we'll cover:

1. **Types of Recommendation Systems**
2. **Popular Algorithms and Techniques**
3. **Evaluation Metrics for Recommendation Systems**
4. **Project: Building a Recommendation System**
5. **Advanced Techniques in Recommendation Systems**

---

## **1. Types of Recommendation Systems**

There are several types of recommendation systems, each suited for different scenarios:

### **1.1. Collaborative Filtering (CF)**
Collaborative Filtering is based on user behavior and similarities between users or items. It assumes that users who agreed on the past will agree again in the future.

- **User-Based CF**: Recommends items based on the preferences of similar users.
- **Item-Based CF**: Recommends items that are similar to the items a user has liked in the past.

### **1.2. Content-Based Filtering**
This approach recommends items based on their content and the user’s profile. For example, if a user has liked action movies, the system recommends other action movies based on genre, actors, or directors.

### **1.3. Hybrid Recommendation Systems**
Hybrid systems combine multiple approaches (e.g., collaborative filtering + content-based filtering) to provide more robust and accurate recommendations.

### **1.4. Knowledge-Based Systems**
These systems use domain-specific knowledge about users and items to recommend products. For example, recommending a vacation package based on user preferences and constraints.

---

## **2. Popular Algorithms and Techniques**

### **2.1. Collaborative Filtering**

1. **Matrix Factorization** (e.g., SVD, ALS)
   - Matrix factorization decomposes the user-item interaction matrix into lower-dimensional latent factors, representing user and item features.
   - Algorithms like **Singular Value Decomposition (SVD)** and **Alternating Least Squares (ALS)** are commonly used for this.

2. **Neighborhood-Based Methods** (User-User or Item-Item)
   - These methods calculate similarity scores (e.g., cosine similarity, Pearson correlation) between users or items and make predictions based on neighbors.

### **2.2. Content-Based Filtering**

- Uses item features (e.g., genre, tags, keywords) and matches them with the user’s profile (e.g., history of likes).
- **TF-IDF** or **word embeddings** (e.g., Word2Vec) are often used for text-based content representation.

### **2.3. Hybrid Methods**

- **Weighted Hybrid**: Combines predictions from collaborative and content-based models by assigning weights to each method.
- **Model-Based Hybrid**: Incorporates both collaborative and content-based features into a unified model (e.g., neural collaborative filtering).

### **2.4. Deep Learning Approaches**

- **Neural Collaborative Filtering (NCF)**: Uses neural networks to model complex interactions between users and items.
- **Autoencoders**: Used for dimensionality reduction and filling missing entries in the user-item matrix.
- **Sequence-Based Models**: RNNs and Transformers can model user behavior sequences for session-based recommendations.

---

## **3. Evaluation Metrics for Recommendation Systems**

Measuring the performance of a recommendation system is crucial. Common evaluation metrics include:

- **Precision@K**: The fraction of relevant items in the top K recommended items.
- **Recall@K**: The fraction of relevant items recovered in the top K recommendations.
- **Mean Absolute Error (MAE)** / **Root Mean Squared Error (RMSE)**: Measures the difference between the predicted and actual ratings in rating-based systems.
- **Hit Rate (HR)**: The percentage of users who receive at least one relevant recommendation.
- **NDCG (Normalized Discounted Cumulative Gain)**: Evaluates the ranking quality by taking the order of items into account.

---

## **4. Project: Building a Movie Recommendation System**

Let’s go through a practical example where we build a **movie recommendation system** using the **MovieLens dataset**, which contains user ratings for various movies.

**Goal**: Recommend movies to users based on their preferences using collaborative filtering and matrix factorization.

**Dataset**: The **MovieLens 100K dataset** contains 100,000 ratings from 943 users on 1,682 movies. It’s publicly available and can be downloaded from [MovieLens](https://grouplens.org/datasets/movielens/).

---

## **Step-by-Step Solution**

### **1. Data Preprocessing**
- Load the dataset, and convert it into a user-item matrix where rows represent users, columns represent movies, and values are ratings.
- Normalize ratings if necessary (e.g., subtract the mean rating for each user).

### **2. Exploratory Data Analysis (EDA)**
- Analyze the distribution of ratings, the sparsity of the user-item matrix, and identify the most-rated movies and the most active users.

### **3. Matrix Factorization using SVD (Singular Value Decomposition)**

SVD decomposes the user-item matrix $R$ into three matrices:
$$
R \approx U \Sigma V^T
$$
Where:
- $U$ represents the user feature matrix.
- $\Sigma$ is a diagonal matrix of singular values.
- $V^T$ is the item feature matrix.

### **4. Implementing the Recommendation System**

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

# Load dataset
ratings = pd.read_csv('movielens-100k/ratings.csv')
ratings = pd.read_csv(
    'https://github.com/MohammedAlawami/Movielens-Dataset/raw/refs/heads/master/datasets/ratings.dat',
    sep='::',
    names=['userId','movieId','rating', 'Timestamp'],
    parse_dates=['Timestamp'],
    engine='python',
    header=None
)

# Create user-item matrix
user_item_matrix_df = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_item_matrix = user_item_matrix_df.values

# Normalize user-item matrix
user_ratings_mean = np.mean(user_item_matrix, axis=1)
R_demeaned = user_item_matrix - user_ratings_mean.reshape(-1, 1)

# Apply SVD
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# Reconstruct the user-item matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix_df.columns)

# Recommend movies for a specific user
def recommend_movies(user_id, num_recommendations=5):
    user_row_number = user_id - 1  # Adjust for zero-indexing
    sorted_user_ratings = predicted_ratings_df.iloc[user_row_number].sort_values(ascending=False)
    recommended_movie_ids = sorted_user_ratings.index[:num_recommendations]
    return recommended_movie_ids

# Example: Recommend movies for user 1
recommend_movies(user_id=1, num_recommendations=5)
```

### **5. Evaluation**
- Split the data into a training set and a test set.
- Evaluate the model using **RMSE** to see how well it predicts unseen ratings.
- Additionally, use metrics like **Precision@K** and **Recall@K** to measure the relevance of the top recommended items.

---

## **5. Advanced Techniques in Recommendation Systems**

### **1. Neural Collaborative Filtering (NCF)**
- **NCF** models user-item interactions using neural networks to capture complex, non-linear patterns.
- Combines user and item embeddings and feeds them into a multi-layer perceptron (MLP) for prediction.

### **2. Autoencoders for Collaborative Filtering**
- **Autoencoders** can be used to reconstruct the user-item matrix, filling in missing values (unrated items) by learning a low-dimensional representation of users and items.

### **3. Sequence-Based and Context-Aware Recommendations**
- **Recurrent Neural Networks (RNNs)** and **Transformers** can model user behavior sequences (e.g., click streams) to make real-time, session-based recommendations.
- **Context-Aware Recommender Systems** consider additional information like time of day, location, and user mood to personalize recommendations further.

### **4. Hybrid Systems**
- Build hybrid models that integrate collaborative filtering, content-based filtering, and deep learning approaches to maximize performance and flexibility.
- For instance, use a weighted combination of content similarity scores and collaborative filtering scores to make recommendations that balance both user behavior and item characteristics.

---

## **Example: Hybrid Recommendation System (Collaborative + Content-Based)**

```python
# Example combining collaborative filtering (matrix factorization) and content-based filtering

def hybrid_recommend(user_id, num_recommendations=5):
    # Step 1: Collaborative Filtering Score
    collaborative_scores = predicted_ratings_df.iloc[user_id - 1]
    
    # Step 2: Content-Based Score
    # (Assume we have a function content_based_score that returns a similarity score for each movie)
    content_scores = content_based_score(user_id)
    
    # Step 3: Weighted Combination
    combined_score = 0.6 * collaborative_scores + 0.4 * content_scores
    
    # Step 4: Get top recommendations
    recommended_movie_ids = combined_score.sort_values(ascending=False).index[:num_recommendations]
    return recommended_movie_ids

# Example usage
hybrid_recommend(user_id=1, num_recommendations=5)
```

---

## 6. **Deep Learning-Based Approaches for Recommendation Systems**

Deep learning-based recommendation systems have become highly popular due to their ability to model complex, non-linear interactions between users and items. These systems leverage deep neural networks to capture hidden patterns, model sequential behaviors, and handle large-scale data efficiently. In this section, we'll explore several deep learning approaches used in recommendation systems, including:

1. **Neural Collaborative Filtering (NCF)**
2. **Autoencoders for Collaborative Filtering**
3. **Deep Learning for Content-Based Recommendations**
4. **Sequential Models (RNNs and Transformers)**
5. **Advanced Techniques (Factorization Machines, Attention Mechanisms)**

For each method, we'll describe its purpose, how it works, and provide practical examples with code.

---

### **1. Neural Collaborative Filtering (NCF)**

#### **Overview**:
Neural Collaborative Filtering (NCF) is a deep learning framework that generalizes traditional **Matrix Factorization (MF)** by replacing the dot product used in MF with a deep neural network that can model complex interactions between users and items.

#### **How It Works**:
- **Embedding Layers**: Each user and item is represented as an embedding vector (latent factors), similar to matrix factorization.
- **Multi-Layer Perceptron (MLP)**: The user and item embeddings are concatenated and fed into a deep neural network (MLP). This allows the model to learn non-linear patterns in user-item interactions.
- **Output Layer**: The final layer outputs a predicted rating or ranking score, depending on the task.

#### **Architecture**:

- **User Embedding**: Maps user IDs to a dense vector (embedding).
- **Item Embedding**: Maps item IDs to a dense vector (embedding).
- **Concatenation**: The embeddings are concatenated, and the result is passed through multiple fully connected layers.
- **Prediction**: A final layer outputs the prediction (e.g., rating or relevance score).

---

#### **Code Example: Neural Collaborative Filtering in PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, hidden_layers):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential()
        input_size = embedding_size * 2  # user + item embedding concatenated
        for hidden_size in hidden_layers:
            self.fc_layers.add_module(f"fc{hidden_size}", nn.Linear(input_size, hidden_size))
            self.fc_layers.add_module(f"relu{hidden_size}", nn.ReLU())
            input_size = hidden_size
        
        # Output layer (for regression tasks)
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_id, item_id):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        x = torch.cat([user_embedding, item_embedding], dim=-1)  # Concatenate embeddings
        
        x = self.fc_layers(x)  # Pass through MLP
        output = self.output(x)  # Output the final prediction
        return output

# Example usage:
num_users = 1000
num_items = 1500
embedding_size = 50
hidden_layers = [128, 64, 32]

# Initialize the model
model = NCF(num_users, num_items, embedding_size, hidden_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # For rating prediction tasks

# Training loop (simplified)
user_ids = torch.LongTensor([0, 1, 2])
item_ids = torch.LongTensor([10, 20, 30])
ratings = torch.FloatTensor([5.0, 4.0, 3.0])

for epoch in range(10):
    optimizer.zero_grad()
    predictions = model(user_ids, item_ids)
    loss = criterion(predictions.squeeze(), ratings)
    loss.backward()
    optimizer.step()
```

#### **Advantages of NCF**:
- **Non-linearity**: NCF can model non-linear interactions between users and items, unlike traditional matrix factorization that relies on linear dot products.
- **Flexibility**: It can easily be extended to handle additional side information (e.g., user demographics or item features).

---

### **2. Autoencoders for Collaborative Filtering**

#### **Overview**:
Autoencoders are used to reconstruct the input data by learning a compressed latent representation of the user-item interactions. This is useful for collaborative filtering tasks, where the autoencoder learns to predict missing values in the user-item matrix.

#### **How It Works**:
- **Encoder**: Maps the user-item interaction vector (e.g., ratings) to a compressed latent space.
- **Decoder**: Reconstructs the user-item interaction from the latent space, filling in missing ratings.
- **Loss**: The reconstruction loss (typically mean squared error) is minimized during training.

#### **Architecture**:
- **Input Layer**: A vector representing a user's ratings for all items (with missing entries as zeros or placeholders).
- **Hidden Layers**: Dense layers that gradually compress the input.
- **Output Layer**: A reconstructed vector of ratings.

---

#### **Code Example: Autoencoder for Collaborative Filtering**

```python
class Autoencoder(nn.Module):
    def __init__(self, num_items):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, num_items),
            nn.Sigmoid(),  # Output is the reconstructed ratings vector
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Example usage:
num_items = 1500
model = Autoencoder(num_items)

# Example input: a user rating vector (1500 items, some ratings missing)
user_ratings = torch.FloatTensor([4.0, 0.0, 0.0, 5.0, ..., 0.0])

# Forward pass
reconstructed_ratings = model(user_ratings)
```

#### **Advantages of Autoencoders**:
- **Dimensionality Reduction**: Autoencoders reduce the dimensionality of the user-item matrix, making it easier to model complex interactions.
- **Filling Missing Values**: Autoencoders are effective for filling in missing ratings in collaborative filtering systems.

---

### **3. Deep Learning for Content-Based Recommendations**

In content-based recommendation systems, deep learning models such as **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)** are used to extract features from raw data like images, text, or audio to make recommendations.

#### **How It Works**:
- **Feature Extraction**: CNNs and RNNs are used to extract features from content (e.g., movie descriptions, user reviews, product images).
- **Content Matching**: These extracted features are matched with the user’s profile or historical preferences to recommend similar items.

#### **Architecture**:
- **Text-Based Content**: Use **RNNs** or **Transformer-based models** like **BERT** for text-based features (e.g., descriptions, reviews).
- **Image-Based Content**: Use **CNNs** (e.g., **ResNet**) for image-based features (e.g., product images, movie posters).

#### **Example**: Movie Recommendation Using Content Descriptions and User Ratings

```python
from transformers import BertTokenizer, BertModel

# Load a pre-trained BERT model for content-based text recommendations
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example movie description
description = "A young lion prince is cast out of his pride by his cruel uncle."

# Tokenize and get the embeddings from BERT
inputs = tokenizer(description, return_tensors='pt')
outputs = model(**inputs)

# Get the pooled output for the entire description (text embedding)
text_embedding = outputs.pooler_output
```

#### **Advantages of Content-Based Deep Learning**:
- **Rich Feature Representation**: Deep learning models can extract complex, high-dimensional features from raw content (e.g., text, images).
- **Scalability**: Can handle a large variety of data types, including text, images, and video.

---

### **4. Sequential Models (RNNs and Transformers)**

#### **Overview**:
Sequential models like **Recurrent Neural Networks (RNNs)** and **Transformers** are used for session-based recommendations, where user interactions are modeled as sequences over time. These models are particularly effective in applications like **next-item prediction** (e.g., recommending the next song in a playlist or product in a session).

#### **How It Works**:
- **RNNs/GRUs/LSTMs**: Capture sequential patterns in user behavior by passing hidden states across time steps.
- **Transformers**: Use self-attention mechanisms to model dependencies between user actions at different points in time, without needing sequential processing like RNNs.

#### **Architecture**:
- **Input**: A sequence of user interactions (e.g., item IDs, timestamps).
- **RNN/Transformer Layers**: Process the sequence to predict the next item

 the user is likely to interact with.
- **Output**: A ranking of items for the next interaction.

#### **Code Example: Sequential Model for Session-Based Recommendation Using RNNs**

```python
class RNNRecModel(nn.Module):
    def __init__(self, num_items, embedding_size, hidden_size):
        super(RNNRecModel, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_items)

    def forward(self, sequence):
        embedded = self.item_embedding(sequence)  # Embed the item sequence
        rnn_output, _ = self.rnn(embedded)  # Pass through RNN
        final_output = self.fc(rnn_output[:, -1, :])  # Use the last output for prediction
        return final_output

# Example usage
num_items = 1500
embedding_size = 50
hidden_size = 128
model = RNNRecModel(num_items, embedding_size, hidden_size)

# Example sequence of item interactions
item_sequence = torch.LongTensor([[10, 20, 30, 40]])  # User's interaction history

# Forward pass to predict the next item
next_item_prediction = model(item_sequence)
```

#### **Advantages of Sequential Models**:
- **Captures Temporal Patterns**: Sequential models can capture the order of user interactions, making them ideal for session-based recommendations.
- **Highly Accurate for Next-Item Prediction**: Models like Transformers can learn complex temporal dependencies and provide highly accurate next-item predictions.

---

### **5. Advanced Techniques**

#### **Factorization Machines with Deep Learning (DeepFM)**:
- **Factorization Machines (FM)** capture pairwise feature interactions. DeepFM extends this by combining factorization machines with deep neural networks to model both low-order and high-order interactions.

#### **Attention Mechanisms**:
- **Attention-based models** can be used to focus on important features or user interactions, improving the recommendation quality. **Self-attention** (used in Transformers) allows the model to weigh different items in a sequence differently, depending on their importance.

---

### **Conclusion**

Deep learning-based approaches have significantly improved recommendation systems by enabling models to capture more complex user-item interactions, learn from various data types (e.g., text, images, sequences), and model dynamic user behavior. Techniques such as **Neural Collaborative Filtering (NCF)**, **Autoencoders**, **RNNs**, and **Transformers** offer flexible and scalable solutions for various recommendation tasks.

