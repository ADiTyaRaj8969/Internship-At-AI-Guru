import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

#   Gaussian Naïve Bayes
def gaussian_nb_demo():
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = GaussianNB()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    print("\n=== Gaussian Naïve Bayes (Iris) ===")
    print("Accuracy:", accuracy_score(y_te, preds))
    print(classification_report(y_te, preds, digits=3))
    
# Build a tiny text dataset

def get_text_corpus():
    corpus = [
        "Win cash now", "Limited offer, buy now",
        "Hi Mom, call me later", "Meeting schedule attached",
        "Claim your free prize", "Project deadline tomorrow"
    ]
    labels = np.array([1, 1, 0, 0, 1, 0])  # 1 = spam, 0 = ham
    return corpus, labels

#   Multinomial Naïve Bayes

def multinomial_nb_demo():
    corpus, labels = get_text_corpus()
    vec = CountVectorizer()                   # counts
    X = vec.fit_transform(corpus)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.33, random_state=0, stratify=labels)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    print("\n=== Multinomial Naïve Bayes (Text counts) ===")
    print("Accuracy:", accuracy_score(y_te, preds))
    print(classification_report(y_te, preds, digits=3))

#   Bernoulli Naïve Bayes

def bernoulli_nb_demo():
    corpus, labels = get_text_corpus()
    vec = CountVectorizer(binary=True)        # 0/1 presence
    X = vec.fit_transform(corpus)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.33, random_state=0, stratify=labels)

    model = BernoulliNB(alpha=1.0)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    print("\n=== Bernoulli Naïve Bayes (Binary presence) ===")
    print("Accuracy:", accuracy_score(y_te, preds))
    print(classification_report(y_te, preds, digits=3))


if __name__ == "__main__":
    gaussian_nb_demo()
    multinomial_nb_demo()
    bernoulli_nb_demo()
