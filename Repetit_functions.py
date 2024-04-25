import pandas as pd

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, make_scorer, fbeta_score

from lightgbm import LGBMClassifier

RANDOM_STATE = 1
import random

random.seed(RANDOM_STATE)


class Repetit:
    def __init__(self):
        pass

    class Paths:

        """
        Класс для доступа к адресам учителей
        """

        teachers_info = "../datasets/teachers_info.feather"
        teachers = "../datasets/teachers.feather"
        lesson_course = "../datasets/lesson_course.feather"
        lessons = "../datasets/lessons.feather"
        teacher_prices = "../datasets/teacher_prices.feather"
        orders = "../datasets/orders.feather"

    paths = Paths()

    def df_info(self, path_df, info=True, describe=True):

        """
        Функция загрузки данных и вывода информации
        """

        df = pd.read_feather(getattr(self.paths, path_df))
        display(df.head(5))
        if info:
            display(df.info())
        if describe:
            display(df.describe())
        return df

    def calculate_unpaid_percentage(self, amount_paid):
        unpaid_count = amount_paid.isnull().sum()
        total_count = len(amount_paid)

        return (unpaid_count / total_count) * 100

    def train_model(self, name_model, model, X, y):

        """
        Функция кросс-валидации модели и вывода результатов
        """

        stratified_kfold = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )
        results = cross_validate(model, X, y, cv=stratified_kfold, scoring="f1")
        display(
            Markdown(
                f'f1 модели {name_model}: **{results["test_score"].mean().round(4)}**'
            )
        )

    def train_models(self, X, y):

        """
        Функция для применения кросс-валидации к трем моделям
        """

        model_LR = LogisticRegression(
            random_state=RANDOM_STATE, class_weight="balanced"
        )
        model_RFC = RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        )
        model_LGBM = LGBMClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", verbose=0
        )

        models = {
            "model_LR": model_LR,
            "model_RFC": model_RFC,
            "model_LGBM": model_LGBM,
        }

        for key, value in models.items():
            self.train_model(key, value, X, y)

    def heatmap(self, data, method='spearman'):
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            data.corr(method=method),
            vmin=-1,
            vmax=1,
            linewidths=1,
            annot=True,
            annot_kws={"fontsize": 12, "fontweight": "bold"},
            fmt=".2f",
        )

    def show_imporatance(self, names, coef):

        """
        Функция построения графика важности признаков
        """

        plt.figure(figsize=(10, len(names) * 0.5))

        df = pd.DataFrame({"names": names, "coef": coef})
        df = df.sort_values(by="coef", ascending=False)
        names_sorted = df["names"]
        coef_sorted = df["coef"]

        sbp = sns.barplot(
            y=names_sorted, x=abs(coef_sorted), orient="h", palette="husl"
        )
        bar_values = [rect.get_width() for rect in sbp.patches]

        for rect, value in zip(sbp.patches, bar_values):
            if value > max(bar_values) * 0.8:
                sbp.annotate(
                    f"{value:.5f}",
                    (value, rect.get_y() + rect.get_height() / 2),
                    xytext=(-10, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                )
            else:
                sbp.annotate(
                    f"{value:.5f}",
                    (value, rect.get_y() + rect.get_height() / 2),
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                )

        plt.ylabel(" ")
        plt.title("Важность признаков")
        plt.tight_layout
        plt.show()

    def my_score(self, X, y):

        """
        Функция скоринга для SelectKBest
        """

        return mutual_info_classif(X, y, random_state=RANDOM_STATE)

    def fbeta_func(self, y_true, y_pred):

        """
        Функция описания fbeta метрики
        """

        fbeta_score = self.fbeta_score(y_true, y_pred, beta=2)
        return fbeta_score

    def grid_search_results(self, grid):

        """
        Функция вывода информации о результатах обучения grid
        """

        display(
            Markdown(
                f"Время обучения лучшей модели: **{round(grid.refit_time_, 2)}** секунды"
            )
        )
        display(
            Markdown(
                f'f1 на кросс валидации: **{grid.cv_results_["mean_test_f1"].mean():.4}**'
            )
        )
        display(
            Markdown(
                f'fb на кросс валидации: **{grid.cv_results_["mean_test_fb_score"].mean():.4}**'
            )
        )
        display(Markdown(f"Наилучший набор параметров: {grid.best_params_}"))

    def fbeta_func(self, y_true, y_pred):

        """
        Функция описания fbeta метрики
        """

        fb_score = fbeta_score(y_true, y_pred, beta=2)
        return fb_score

    def my_fb_scorer(self):

        """
        Функция создания оценщика
        """

        return make_scorer(self.fbeta_func)

    def confusion_matrix_heatmap(self, y_real, y_pred):

        """
        Функция построяния heatmap матрицы ошибок
        """

        labels = [0, 1]
        mcm = confusion_matrix(y_real, y_pred, labels=labels)
        plt.figure(figsize=(7, 5))
        sns.heatmap(
            mcm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Предсказанные значения", fontsize=14)
        plt.ylabel("Реальные значения", fontsize=14)
        plt.title("Матрица ошибок", fontsize=18)
        plt.show()