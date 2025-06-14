import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred):
    """Вывод матрицы ошибок"""
    classes = ["cart_request",
               "clear_cart_request",
               "complete_order_request",
               "goodbye",
               "greeting",
               "menu_request",
               "order_request",
               "price_request"]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 12))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Действительный класс')
    plt.title('Матрица ошибок')
    plt.show()


def plot_learning_curve(estimator, X, y):
    """Вывод графика кривой обучения"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(.1, 1.0, 5))

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Обучающий счет")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Тестовый счет")
    plt.xlabel("Кол-во примеров в обучающей выборке")
    plt.ylabel("Точность")
    plt.legend()
    plt.title("Кривая обучения")
    plt.show()
