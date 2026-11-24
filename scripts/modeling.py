"""Models and metrics shared by the modelling, robustness and explanation stages."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn

from common import PROCESSED, TARGET_LEVELS

INNER_GROUP = {"leave_period_out": "cooler_period", "block_latin": "accumulator_block"}


class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden, dropout):
        super().__init__()
        layers, size = [], n_inputs
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
            size = width
        layers.append(nn.Linear(size, n_outputs))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


class TorchMLP:
    """Small MLP. The scaler is fitted inside `fit`, on training data only."""

    def __init__(self, hidden, dropout, learning_rate, weight_decay, epochs, batch_size, threads, seed=42):
        self.config = dict(hidden=hidden, dropout=dropout)
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.epochs, self.batch_size, self.threads, self.seed = epochs, batch_size, threads, seed

    def _scale(self, X):
        return torch.tensor((np.asarray(X, dtype=np.float32) - self.mean_) / self.scale_, dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        generator = torch.Generator().manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.classes_ = np.unique(y)
        targets = torch.tensor(np.searchsorted(self.classes_, y), dtype=torch.long)
        inputs = self._scale(X)
        self.net_ = Network(X.shape[1], len(self.classes_), **self.config)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.CrossEntropyLoss()
        self.net_.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(inputs), generator=generator)
            for start in range(0, len(inputs), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < 2:
                    continue
                optimizer.zero_grad()
                loss_function(self.net_(inputs[batch]), targets[batch]).backward()
                optimizer.step()
        self.net_.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.softmax(self.net_(self._scale(X)), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def make_lightgbm(fixed, tuned, seed=42):
    return lgb.LGBMClassifier(random_state=seed, verbose=-1, **fixed, **tuned)


def make_mlp(config, seed=42):
    return TorchMLP(seed=seed, **config)


def make_logreg(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def make_shallow_tree(depth, seed=42):
    return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=seed)


def make_dummy():
    return DummyClassifier(strategy="prior")


def score(y_true, y_pred, probabilities, classes, target):
    classes = list(classes)
    precision = []
    for level in np.unique(y_true):
        column = probabilities[:, classes.index(level)] if level in classes else np.zeros(len(y_true))
        precision.append(average_precision_score(y_true == level, column))
    rank = {level: index for index, level in enumerate(TARGET_LEVELS[target])}
    true_rank = np.array([rank[v] for v in y_true])
    pred_rank = np.array([rank[v] for v in y_pred])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_pr_auc": float(np.mean(precision)),
        "ordinal_mae": float(np.abs(true_rank - pred_rank).mean()),
    }


def inner_folds(labels, train, scheme, y, seed):
    """Two inner folds that mirror the outer scheme, so tuning cannot use held-out structure."""
    if scheme in INNER_GROUP:
        groups = labels[INNER_GROUP[scheme]].to_numpy()[train]
        splitter = GroupKFold(n_splits=2, shuffle=True, random_state=seed)
        return list(splitter.split(train, groups=groups))
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    return list(splitter.split(train, y[train]))


def tune_lightgbm(X, y, labels, train, scheme, fixed, grid, target, seed):
    results = []
    folds = inner_folds(labels, train, scheme, y, seed)
    for tuned in grid:
        values = []
        for fit_index, valid_index in folds:
            fit_rows, valid_rows = train[fit_index], train[valid_index]
            model = make_lightgbm(fixed, tuned, seed).fit(X[fit_rows], y[fit_rows])
            prediction = model.predict(X[valid_rows])
            values.append(f1_score(y[valid_rows], prediction, average="macro"))
        results.append(float(np.mean(values)))
    return grid[int(np.argmax(results))], results


def load_processed():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    folds = pd.read_parquet(PROCESSED / "folds.parquet")
    return features, labels, folds


def fold_indices(folds, scheme, fold):
    part = folds[(folds["scheme"] == scheme) & (folds["fold"] == fold)]
    return (
        part.loc[part["role"] == "train", "cycle"].to_numpy(),
        part.loc[part["role"] == "test", "cycle"].to_numpy(),
    )
