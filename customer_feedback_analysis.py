import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Constants
SERVICE_KEYWORDS = ['delivery', 'shipping', 'packaging', 'customer service', 'support', 
                   'return policy', 'response time', 'handling', 'dispatch', 'arrival time',
                   'seller communication', 'order processing', 'refund process', 'timely',
                   'professional service']
QUALITY_KEYWORDS = ['durable', 'material', 'build quality', 'performance', 'functionality',
                   'design', 'appearance', 'longevity', 'reliability', 'sturdiness',
                   'workmanship', 'craftsmanship', 'finish', 'texture', 'color accuracy']
UNDESIRED_KEYWORDS = ['unrelated', 'spam', 'advertisement', 'personal opinion', 'not relevant',
                     'wrong product', 'mistake', 'accident', 'not about product', 'off topic',
                     'general comment', 'no details', 'vague', 'unclear', 'irrelevant']

def load_data(filepath):
    """Load and preprocess the dataset"""
    # Try both possible file naming conventions
    try:
        df = pd.read_csv('amazon_reviews.csv')
    except FileNotFoundError:
        df = pd.read_csv('amazon_review.csv')
    
    # Map column names with case-insensitive matching
    column_map = {}
    for col in df.columns:
        if 'amazonid' in col.lower():
            column_map[col] = 'article_id'
        elif 'overall' in col.lower():
            column_map[col] = 'rating'
        elif 'reviewtext' in col.lower():
            column_map[col] = 'review_comment'
    
    df = df.rename(columns=column_map)
    df = df[['article_id', 'rating', 'review_comment']].copy()
    df['review_comment'] = df['review_comment'].fillna('')
    # Convert rating to numeric if needed
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    return df.dropna(subset=['rating'])

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def classify_feedback(text):
    """Classify feedback into service, quality, or undesired"""
    text = clean_text(text)
    
    # Check for undesired feedback first
    undesired_score = sum(text.count(word) for word in UNDESIRED_KEYWORDS)
    if undesired_score > 0:
        return None, None
    
    # Classify service and quality
    service_score = sum(text.count(word) for word in SERVICE_KEYWORDS)
    quality_score = sum(text.count(word) for word in QUALITY_KEYWORDS)
    
    # Use sentiment analysis as tie-breaker
    sentiment = TextBlob(text).sentiment.polarity
    
    service = 1 if (service_score > 0 and sentiment >= 0) or (service_score == 0 and sentiment > 0.2) else 0
    quality = 1 if (quality_score > 0 and sentiment >= 0) or (quality_score == 0 and sentiment > 0.2) else 0
    
    return service, quality

def determine_final_classification(row):
    """Determine final classification based on rules"""
    rating = row['rating']
    service = row['service']
    quality = row['quality']
    
    if service == 1 and quality == 1:
        if rating >= 4:
            return 'BUY'
        elif rating == 3:
            return 'BUY'
        else:
            return 'BAD BUY'
    elif service == 0 and quality == 0:
        if rating <= 2:
            return 'NOT BUY'
        elif rating == 3:
            return 'NOT BUY'
        else:
            return 'NOT BUY'
    else:
        # Mixed feedback case
        if (rating >= 4 and (service == 0 or quality == 0)) or (rating <= 2 and (service == 1 or quality == 1)):
            if (service == 1 and quality == 0) or (service == 0 and quality == 1):
                return 'POTENTIALLY GOOD BUY'
        return 'NOT BUY'

def prepare_features(df):
    """Prepare features for model training"""
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df['review_comment'])
    
    # Combine with other features
    X_other = df[['rating', 'service', 'quality']].values
    X = np.hstack((X_text.toarray(), X_other))
    
    # Target variable
    y = df['final_classification'].map({'BUY':0, 'NOT BUY':1, 'BAD BUY':2, 'POTENTIALLY GOOD BUY':3})
    
    return X, y, vectorizer

def train_model(X, y):
    """Train and evaluate Random Forest model with class weighting"""
    # Calculate class weights
    class_counts = np.bincount(y)
    class_weights = {i: sum(class_counts)/class_counts[i] for i in range(len(class_counts))}
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weights
    )
    
    # Enhanced 10-fold stratified cross-validation
    from sklearn.model_selection import StratifiedKFold
    
    # Create stratified 5-fold cross-validator
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Cross-validation with multiple metrics
    scoring = {
        'balanced_accuracy': 'balanced_accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    
    cv_results = cross_validate(
        model, X, y, 
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Print comprehensive CV results with zero_division
    print("\nEnhanced 10-Fold Cross-Validation Results:")
    print(f"Mean Balanced Accuracy: {np.mean(cv_results['test_balanced_accuracy']):.2f} ± {np.std(cv_results['test_balanced_accuracy']):.2f}")
    print(f"Mean Precision (Macro): {np.mean(cv_results['test_precision_macro']):.2f} ± {np.std(cv_results['test_precision_macro']):.2f}")
    print(f"Mean Recall (Macro): {np.mean(cv_results['test_recall_macro']):.2f} ± {np.std(cv_results['test_recall_macro']):.2f}")
    print(f"Mean F1 (Macro): {np.mean(cv_results['test_f1_macro']):.2f} ± {np.std(cv_results['test_f1_macro']):.2f}")
    
    # Plot feature importance
    model.fit(X, y)  # Fit on full data for feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    plt.figure(figsize=(10,6))
    plt.title("Top 20 Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    # Plot learning curve
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=skf, scoring='balanced_accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.close()
    
    # Leave-One-Out Cross-Validation
    from sklearn.model_selection import LeaveOneOut
    from sklearn.dummy import DummyClassifier
    
    # Majority class baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy_scores = cross_val_score(dummy, X, y, cv=LeaveOneOut(), scoring='balanced_accuracy')
    print(f"\nMajority Class Baseline (Balanced Accuracy): {np.mean(dummy_scores):.2f}")
    
    # Convert y to numpy array to avoid pandas indexing issues
    y_array = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
    
    # Actual model evaluation
    loo = LeaveOneOut()
    y_pred = []
    y_true = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        
        # Apply SMOTE only to training data
        X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        
        model.fit(X_train_res, y_train_res)
        # Collect predictions and true labels
        y_pred.append(model.predict(X_test)[0])  # Single prediction
        y_true.append(y_test[0])  # Single true label
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Ensure we have matching sample sizes
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Confusion matrix with labels (using aligned y_true and y_pred)
    class_names = ['BUY', 'NOT BUY', 'BAD BUY', 'POTENTIALLY GOOD BUY']
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names[:len(np.unique(y))],
                yticklabels=class_names[:len(np.unique(y))])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 features
    plt.figure(figsize=(10,6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig('feature_importances.png')
    plt.close()
    
    return model

def visualize_sentiment(df):
    """Visualize sentiment scores of reviews"""
    df['sentiment'] = df['review_comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df['sentiment'], bins=20, kde=True)
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='--')  # Line at neutral sentiment
    plt.savefig('sentiment_distribution.png')
    plt.close()

def visualize_results(df):
    """Create visualizations of the results"""
    # Rating distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x='rating', data=df)
    plt.title('Rating Distribution')
    plt.savefig('rating_distribution.png')
    plt.close()
    
    # Classification distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x='final_classification', data=df)
    plt.title('Final Classification Distribution')
    plt.savefig('classification_distribution.png')
    plt.close()
    
    # Service vs Quality
    plt.figure(figsize=(10, 5))
    sns.countplot(x='service', hue='quality', data=df)
    plt.title('Service vs Quality Feedback')
    plt.savefig('service_vs_quality.png')
    plt.close()

def main():
    # Load and preprocess data
    df = load_data('amazon_reviews.csv')
    df['review_comment'] = df['review_comment'].apply(clean_text)
    
    # Classify feedback
    classifications = df['review_comment'].apply(classify_feedback)
    df[['service', 'quality']] = pd.DataFrame(classifications.tolist(), index=df.index)
    
    # Remove undesired feedback
    df = df.dropna(subset=['service', 'quality'])
    
    # Determine final classification
    df['final_classification'] = df.apply(determine_final_classification, axis=1)
    
    # Prepare features and train model
    X, y, vectorizer = prepare_features(df)
    model = train_model(X, y)
    
    # Visualize results
    visualize_results(df)
    
    return model, vectorizer

if __name__ == '__main__':
    model, vectorizer = main()