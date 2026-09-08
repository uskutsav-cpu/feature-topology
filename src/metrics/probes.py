import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.exceptions import ConvergenceWarning


def evaluate_probes(train_h, test_h, train_y, test_y, train_phi, test_phi, seed=0,
                    max_iter=300, periodic=True):
    target = (np.stack((np.cos(train_phi), np.sin(train_phi)), 1) if periodic else train_phi[:, None])
    truth = (np.stack((np.cos(test_phi), np.sin(test_phi)), 1) if periodic else test_phi[:, None])
    result = {}
    for name, clf, reg in [
        ("linear", LogisticRegression(max_iter=max_iter), Ridge(alpha=1.)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=max_iter,
                              early_stopping=True, random_state=seed),
         MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=max_iter,
                      early_stopping=True, random_state=seed))]:
        task = make_pipeline(StandardScaler(), clf)
        nuisance = make_pipeline(StandardScaler(), reg)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            task.fit(train_h, train_y)
            nuisance.fit(train_h, target if target.shape[1] > 1 else target.ravel())
        pred = nuisance.predict(test_h).reshape(truth.shape)
        scores = {"task_accuracy": float(accuracy_score(test_y, task.predict(test_h))),
                  "nuisance_r2": float(r2_score(truth, pred)),
                  "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught)}
        if periodic:
            error = np.angle(np.exp(1j*(np.arctan2(pred[:, 1], pred[:, 0])-test_phi)))
            scores.update(angular_mae=float(np.mean(np.abs(error))),
                          angular_cosine=float(np.mean(np.cos(error))))
        else:
            scores["nuisance_mae"] = float(np.mean(np.abs(pred-truth)))
        result[name] = scores
    return result
