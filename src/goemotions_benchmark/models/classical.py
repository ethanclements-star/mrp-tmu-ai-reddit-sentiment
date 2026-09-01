from goemotions_benchmark.labels import EMOTION_NAMES, emotion_index
from goemotions_benchmark.data import GoEmotionsSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def _labels_to_matrix(labels: list[list[str]]) -> list[list[int]]:
    index = emotion_index()
    matrix = []
    for row in labels:
        vec = [0] * len(EMOTION_NAMES)
        for name in row:
            if name in index:
                vec[index[name]] = 1
        matrix.append(vec)
    return matrix


def _matrix_to_labels(matrix: list[list[int]]) -> list[list[str]]:
    labels = []
    for row in matrix:
        active = [EMOTION_NAMES[i] for i, value in enumerate(row) if value == 1]
        labels.append(active)
    return labels


def predict_tfidf_logreg(train: GoEmotionsSplit, test: GoEmotionsSplit) -> list[list[str]]:
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train.texts)
    x_test = vectorizer.transform(test.texts)

    y_train = _labels_to_matrix(train.emotion_labels)
    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, random_state=42),
    )
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    return _matrix_to_labels(y_pred.tolist())
