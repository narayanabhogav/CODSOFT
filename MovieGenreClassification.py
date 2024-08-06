import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the datasets with low_memory=False to avoid DtypeWarning
dataset1 = pd.read_csv('test_data.csv', low_memory=False)
dataset2 = pd.read_csv('test_data_solution.csv', low_memory=False)
dataset3 = pd.read_csv('train_data.csv', low_memory=False)

# Print column names for each dataset to identify issues
print("Dataset1 columns:", dataset1.columns)
print("Dataset2 columns:", dataset2.columns)
print("Dataset3 columns:", dataset3.columns)

# Define the correct column names for each dataset
# Adjust the following lines according to the actual columns in your datasets
dataset1.columns = ['id', 'title', 'plot_summary']
dataset2.columns = ['id', 'title', 'genre', 'plot_summary']
dataset3.columns = ['id', 'title', 'genre', 'plot_summary']

# Combine only the relevant datasets (dataset1, dataset2, dataset3)
data = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)


# Display the first few rows of the combined dataset
print(data.head())

# Identify the correct column names for 'plot_summary' and 'genre'
plot_column = 'plot_summary'
genre_column = 'genre'

if plot_column in data.columns and genre_column in data.columns:
    # Preprocessing
    data[plot_column] = data[plot_column].fillna('')

    # Drop rows with missing genre
    data = data.dropna(subset=[genre_column])

    # Encode the target labels
    label_encoder = LabelEncoder()
    data['encoded_genre'] = label_encoder.fit_transform(data[genre_column])

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    X_train = train_data[plot_column]
    y_train = train_data['encoded_genre']
    X_test = test_data[plot_column]
    y_test = label_encoder.transform(test_data[genre_column])

    # Convert text data to TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Model Building - Naive Bayes Classifier
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
else:
    print(f"Columns '{plot_column}' and/or '{genre_column}' not found in the combined dataset.")
