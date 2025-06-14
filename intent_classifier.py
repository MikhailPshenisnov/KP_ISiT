import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_preparation import get_menu, get_intent_dataset
from model_metrics_visualization import plot_confusion_matrix, plot_learning_curve

from config import MENU_FILE_PATH


class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 3))),
            ("clf", LinearSVC())
        ])

    def train(self, X, y):
        """Обучение модели классификатора"""
        self.pipeline.fit(X, y)

    def predict(self, text):
        """Предсказание намерения для текста"""
        return self.pipeline.predict([text])[0]

    def save(self, file_path):
        """Сохранение модели в файл"""
        with open(file_path, "wb") as file:
            pickle.dump(self.pipeline, file)

    @classmethod
    def load(cls, file_path):
        """Загрузка модели из файла"""
        with open(file_path, "rb") as file:
            pipeline = pickle.load(file)
        classifier = cls()
        classifier.pipeline = pipeline
        return classifier


def prepare_intents_dataset_for_model(file_path):
    """Подготовка датасета намерений для модели"""
    data = get_intent_dataset(file_path, get_menu(MENU_FILE_PATH))

    X, y = [], []

    for intent, intent_data in data["intents"].items():
        for example in intent_data["examples"]:
            X.append(example)
            y.append(intent)

    classes = []
    for cls in y:
        if cls not in classes:
            classes.append(cls)

    return X, y, classes


def train_and_save_model(dataset_file_path, model_file_path):
    """Обучение и сохранение модели, вывод различных метрик модели"""
    X, y, classes = prepare_intents_dataset_for_model(dataset_file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = IntentClassifier()
    classifier.train(X_train, y_train)
    classifier.save(model_file_path)

    y_pred = classifier.pipeline.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    print(f"Точность : {accuracy:.4f}\n")

    print("Отчет по классам:")
    print(f"{classification_report(y_test, y_pred)}\n")

    print("Матрица ошибок будет выведена на графике\n")
    plot_confusion_matrix(y_test, y_pred)

    print("Кривая обучения также будет представлена на графике\n")
    plot_learning_curve(classifier.pipeline, X, y)

    return classifier
