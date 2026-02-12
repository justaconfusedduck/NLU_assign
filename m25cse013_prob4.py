import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SPORT_CATS = ['rec.sport.baseball', 'rec.sport.hockey']
POLITICS_CATS = ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast']

LABEL_MAP = {
    'rec.sport.baseball': 'Sport',
    'rec.sport.hockey': 'Sport',
    'talk.politics.misc': 'Politics',
    'talk.politics.guns': 'Politics',
    'talk.politics.mideast': 'Politics',
}


def load_data():
    print("=" * 60)
    print("STEP 1: Loading Data from 20 Newsgroups")
    print("=" * 60)

    all_cats = SPORT_CATS + POLITICS_CATS
    dataset = fetch_20newsgroups(
        subset='all',
        categories=all_cats,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )

    texts = dataset.data
    target_names = np.array(dataset.target_names)
    raw_labels = target_names[dataset.target]
    labels = np.array([LABEL_MAP[l] for l in raw_labels])

    print(f"Total documents loaded: {len(texts)}")
    print(f"Sport documents:    {np.sum(labels == 'Sport')}")
    print(f"Politics documents: {np.sum(labels == 'Politics')}")
    print()

    return texts, labels


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


def preprocess_all(texts):
    print("STEP 2: Preprocessing Text Data")
    print("-" * 40)
    cleaned = [preprocess_text(doc) for doc in texts]
    print(f"Preprocessed {len(cleaned)} documents.")
    print(f"Sample (first 200 chars): {cleaned[0][:200]}...")
    print()
    return cleaned


def dataset_analysis(texts, labels, cleaned_texts):
    print("STEP 3: Dataset Analysis")
    print("-" * 40)

    label_counts = Counter(labels)
    for lbl, cnt in label_counts.items():
        print(f"  {lbl}: {cnt} documents")

    doc_lengths = [len(doc.split()) for doc in cleaned_texts]
    sport_lengths = [len(cleaned_texts[i].split()) for i in range(len(labels)) if labels[i] == 'Sport']
    politics_lengths = [len(cleaned_texts[i].split()) for i in range(len(labels)) if labels[i] == 'Politics']

    print(f"\n  Average document length (words): {np.mean(doc_lengths):.1f}")
    print(f"  Sport avg length:    {np.mean(sport_lengths):.1f}")
    print(f"  Politics avg length: {np.mean(politics_lengths):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#4C72B0', '#DD8452']
    axes[0].bar(label_counts.keys(), label_counts.values(), color=colors)
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Documents')
    for i, (lbl, cnt) in enumerate(label_counts.items()):
        axes[0].text(i, cnt + 20, str(cnt), ha='center', fontweight='bold')

    axes[1].hist(sport_lengths, bins=30, alpha=0.7, label='Sport', color=colors[0])
    axes[1].hist(politics_lengths, bins=30, alpha=0.7, label='Politics', color=colors[1])
    axes[1].set_title('Document Length Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    all_sport_words = ' '.join([cleaned_texts[i] for i in range(len(labels)) if labels[i] == 'Sport']).split()
    all_politics_words = ' '.join([cleaned_texts[i] for i in range(len(labels)) if labels[i] == 'Politics']).split()

    sport_common = Counter(all_sport_words).most_common(15)
    politics_common = Counter(all_politics_words).most_common(15)

    print(f"\n  Top 10 Sport words:    {[w for w, _ in sport_common[:10]]}")
    print(f"  Top 10 Politics words: {[w for w, _ in politics_common[:10]]}")

    words_s, counts_s = zip(*sport_common)
    words_p, counts_p = zip(*politics_common)

    x_pos = np.arange(15)
    width = 0.35
    axes[2].barh(x_pos - width / 2, counts_s[::-1], width, label='Sport', color=colors[0])
    axes[2].barh(x_pos + width / 2, counts_p[::-1], width, label='Politics', color=colors[1])
    axes[2].set_yticks(x_pos)
    axes[2].set_yticklabels([f"{words_s[::-1][i]} / {words_p[::-1][i]}" for i in range(15)], fontsize=8)
    axes[2].set_title('Top 15 Words per Class', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Frequency')
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'dataset_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved dataset analysis plot to {path}")
    print()

    return {
        'total': len(texts),
        'sport_count': label_counts['Sport'],
        'politics_count': label_counts['Politics'],
        'avg_length': np.mean(doc_lengths),
        'sport_avg_length': np.mean(sport_lengths),
        'politics_avg_length': np.mean(politics_lengths),
    }


def extract_features(X_train_text, X_test_text):
    print("STEP 4: Feature Extraction")
    print("-" * 40)

    bow_vectorizer = CountVectorizer(max_features=10000)
    X_train_bow = bow_vectorizer.fit_transform(X_train_text)
    X_test_bow = bow_vectorizer.transform(X_test_text)
    print(f"  Bag of Words shape:     {X_train_bow.shape}")

    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    print(f"  TF-IDF shape:           {X_train_tfidf.shape}")

    bigram_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train_bigram = bigram_vectorizer.fit_transform(X_train_text)
    X_test_bigram = bigram_vectorizer.transform(X_test_text)
    print(f"  Bigram TF-IDF shape:    {X_train_bigram.shape}")
    print()

    features = {
        'Bag of Words': (X_train_bow, X_test_bow),
        'TF-IDF': (X_train_tfidf, X_test_tfidf),
        'Bigram TF-IDF': (X_train_bigram, X_test_bigram),
    }

    return features


def get_models():
    return {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'SVM (LinearSVC)': LinearSVC(max_iter=10000, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }


def run_experiments(features, y_train, y_test):
    print("STEP 5: Training Models & Evaluating")
    print("=" * 60)

    results = []

    for feat_name, (X_train, X_test) in features.items():
        print(f"\n--- Feature: {feat_name} ---")

        models = get_models()

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label='Sport', average='binary')
            rec = recall_score(y_test, y_pred, pos_label='Sport', average='binary')
            f1 = f1_score(y_test, y_pred, pos_label='Sport', average='binary')

            results.append({
                'Model': model_name,
                'Feature': feat_name,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4),
            })

            print(f"\n  {model_name}:")
            print(f"    Accuracy:  {acc:.4f}")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall:    {rec:.4f}")
            print(f"    F1-Score:  {f1:.4f}")

            cm = confusion_matrix(y_test, y_pred, labels=['Politics', 'Sport'])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Politics', 'Sport'],
                        yticklabels=['Politics', 'Sport'], ax=ax)
            ax.set_title(f'{model_name} + {feat_name}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            safe_name = f"cm_{model_name}_{feat_name}".replace(' ', '_').replace('(', '').replace(')', '')
            path = os.path.join(PLOT_DIR, f'{safe_name}.png')
            plt.savefig(path, dpi=120, bbox_inches='tight')
            plt.close()

    return results


def save_results(results):
    print("\n" + "=" * 60)
    print("STEP 6: Saving Results")
    print("=" * 60)

    df = pd.DataFrame(results)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\n" + "=" * 60)
    print("RESULTS COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    feature_names = df['Feature'].unique()

    for idx, feat in enumerate(feature_names):
        subset = df[df['Feature'] == feat]
        x = np.arange(len(subset))
        width = 0.2

        axes[idx].bar(x - 1.5 * width, subset['Accuracy'], width, label='Accuracy', color='#4C72B0')
        axes[idx].bar(x - 0.5 * width, subset['Precision'], width, label='Precision', color='#55A868')
        axes[idx].bar(x + 0.5 * width, subset['Recall'], width, label='Recall', color='#C44E52')
        axes[idx].bar(x + 1.5 * width, subset['F1-Score'], width, label='F1-Score', color='#8172B2')

        axes[idx].set_title(f'{feat}', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(subset['Model'], rotation=25, ha='right', fontsize=9)
        axes[idx].set_ylim(0.5, 1.05)
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylabel('Score')

    plt.suptitle('Model Performance Comparison Across Feature Representations',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to {path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df = df.pivot(index='Model', columns='Feature', values='Accuracy')
    pivot_df.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
    ax.set_title('Accuracy Comparison: Models × Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Model')
    ax.set_ylim(0.7, 1.02)
    ax.legend(title='Feature', fontsize=9)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison chart saved to {path}")

    return df


def print_detailed_reports(features, y_train, y_test):
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 60)

    for feat_name, (X_train, X_test) in features.items():
        models = get_models()
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"\n{'='*50}")
            print(f" {model_name} + {feat_name}")
            print(f"{'='*50}")
            print(classification_report(y_test, y_pred, target_names=['Politics', 'Sport']))


def main():
    print("\n" + "#" * 60)
    print("#  SPORT vs POLITICS TEXT CLASSIFIER")
    print("#  Using 20 Newsgroups Dataset")
    print("#" * 60 + "\n")

    texts, labels = load_data()

    cleaned_texts = preprocess_all(texts)

    stats = dataset_analysis(texts, labels, cleaned_texts)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train set size: {len(X_train_text)}")
    print(f"Test set size:  {len(X_test_text)}")
    print()

    features = extract_features(X_train_text, X_test_text)

    results = run_experiments(features, y_train, y_test)

    df = save_results(results)

    print_detailed_reports(features, y_train, y_test)

    best = df.loc[df['Accuracy'].idxmax()]
    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(f"  Model:    {best['Model']}")
    print(f"  Feature:  {best['Feature']}")
    print(f"  Accuracy: {best['Accuracy']}")
    print(f"  F1-Score: {best['F1-Score']}")
    print("\nDone! Check the 'plots/' directory for visualizations.")
    print(f"Results table saved to 'results.csv'.")


if __name__ == '__main__':
    main()
