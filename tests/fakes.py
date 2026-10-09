"""Test doubles: a scripted FakeLLM (no model needed) and synthetic datasets (plain / time / group)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from mlagent.llm import RateLimitError

EXP = '''from mlkit import *
{imports}
EXP_ID = "{eid}"
{boom}
df = load_train(); y = get_y()
X = df.drop(columns=[c for c in [TARGET, ID] if c and c in df.columns])
test = load_test(); X_test = test[X.columns] if test is not None else None
def fit_predict(X_tr, y_tr, X_va, X_te):
    A, B, C = simple_prep(X_tr, X_va, X_te)
    m = {model}.fit(A, y_tr)
    return predict_any(m, B), (predict_any(m, C) if C is not None else None)
run_cv(EXP_ID, fit_predict, X, y, X_test)
'''

OPTUNA = '''from mlkit import *
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)
EXP_ID = "{eid}"
df = load_train(); y = get_y()
X = df.drop(columns=[c for c in [TARGET, ID] if c and c in df.columns])
test = load_test(); X_test = test[X.columns] if test is not None else None
def make_fp(params):
    def fit_predict(X_tr, y_tr, X_va, X_te):
        A, B, C = simple_prep(X_tr, X_va, X_te)
        m = HistGradientBoostingClassifier(**params, random_state=SEED).fit(A, y_tr)
        return predict_any(m, B), (predict_any(m, C) if C is not None else None)
    return fit_predict
def objective(trial):
    params = {{"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
               "max_depth": trial.suggest_int("max_depth", 2, 6)}}
    return run_cv(EXP_ID, make_fp(params), X, y, final=False)["cv_mean"]
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=5, timeout=60)
run_cv(EXP_ID, make_fp(study.best_params), X, y, X_test)
'''


class FakeLLM:
    model = "fake-llm"

    def __init__(self, cv=None, api_exp=None, oom_exp=None, oom_until=1, rate_limit_after=None,
                 lightgbm_hyp=False, optuna_tuner=False):
        self.cv = cv or {"kind": "stratified", "n_splits": 4}
        self.api_exp, self.oom_exp, self.oom_until = api_exp, oom_exp, oom_until
        self.limit, self.calls, self.lightgbm_hyp, self.optuna_tuner = rate_limit_after, 0, lightgbm_hyp, optuna_tuner
        self.code_log: list[str] = []

    def _tick(self):
        self.calls += 1
        if self.limit is not None and self.calls > self.limit:
            raise RateLimitError("fake 429")

    def chat(self, system, user, json_mode=False, temperature=None):
        self._tick()
        return "Fake cheat sheet / summary text."

    def json(self, system, user, model, retries=2):
        self._tick()
        n = model.__name__
        if n == "Questions":
            return model(questions=["How balanced is the target?", "Which columns are missing?"])
        if n == "StrategyOut":
            hyps = [{"hypothesis": f"hyp {i}", "change": f"use {100 + 50 * i} trees", "kind": "model",
                     "base": "best", "params": {"n": 100 + 50 * i}} for i in range(4)]
            if self.lightgbm_hyp:
                hyps.insert(0, {"hypothesis": "boosting beats forests", "change": "switch to lightgbm LGBMClassifier",
                                "kind": "model", "base": "best", "params": {"model": "lightgbm"}})
            return model(validation=self.cv, risks=["small data"], domain_notes=["(unverified) title helps"],
                         hypotheses=hyps)
        if n == "Verdict":
            return model(verdict="approve", reasons=[])
        if n == "AnalysisOut":
            return model(bottleneck="variance", evidence="fake", hypotheses=[
                {"hypothesis": f"more trees {self.calls}", "change": f"use {700 + self.calls} trees", "kind": "model", "base": "best"}])
        if n == "DocQuery":
            return model(library="sklearn", objects=["sklearn.ensemble.NoSuchClass" if "NoSuchClass" in user
                                                     else "sklearn.ensemble.RandomForestClassifier"])
        raise AssertionError(n)

    def code(self, system, user, retries=1):
        self._tick()
        name = re.search(r"Script file: (\S+)\.py", user).group(1)
        self.code_log.append(name)
        task = user.split("TASK:")[-1]
        if name.startswith("profile"):
            return 'from mlkit import *\nemit({"answers": {"How balanced is the target?": "about 38% positive"}})\n'
        if name.startswith("val_"):
            return ('from mlkit import *\nemit({"features_used": [], "uses_id_as_feature": False, '
                    '"fit_before_split": False, "suspect_features": [], "notes": "fake"})\n')
        if name.startswith("signals"):
            return 'from mlkit import *\nemit({"per_class_error": {"0": 0.1, "1": 0.3}})\n'
        if name.startswith("t_") and self.optuna_tuner:
            return OPTUNA.format(eid=name)
        boom = ""
        if self.api_exp == name and "## sklearn" not in user:
            boom = 'raise TypeError("__init__() got an unexpected keyword argument \'foo\'")'
        if self.oom_exp == name:
            need = "DOWNGRADE LEVEL 2" if self.oom_until >= 2 else "DOWNGRADE LEVEL"
            if need not in user:
                boom = 'raise MemoryError("Unable to allocate 9.00 GiB for an array")'
        if "lightgbm" in task.lower() and not name.startswith("t_"):
            return EXP.format(eid=name, boom=boom, imports="from lightgbm import LGBMClassifier",
                              model="LGBMClassifier(n_estimators=80, n_jobs=N_JOBS, verbose=-1, random_state=SEED)")
        n = 150 + 25 * (sum(map(ord, name)) % 5)
        return EXP.format(eid=name, boom=boom, imports="from sklearn.ensemble import RandomForestClassifier",
                          model=f"RandomForestClassifier(n_estimators={n}, n_jobs=N_JOBS, random_state=SEED)")


def make_data(folder: Path, kind: str = "plain", n: int = 600) -> Path:
    """Titanic-like data. kind: plain | time (adds `date`, test is later) | group (adds `grp`, test has new groups)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"PassengerId": range(1, n + 1), "Pclass": rng.integers(1, 4, n),
                       "Sex": rng.choice(["male", "female"], n), "Age": rng.normal(30, 12, n).round(1),
                       "Fare": rng.exponential(30, n).round(2), "Embarked": rng.choice(["S", "C", "Q", None], n)})
    df.loc[rng.random(n) < 0.15, "Age"] = np.nan
    z = 1.5 * (df.Sex == "female") - 0.8 * (df.Pclass - 2) - 0.02 * (df.Age.fillna(30) - 30)
    if kind == "time":
        df["date"] = pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
        z = z + np.linspace(-0.5, 0.5, n)
    if kind == "group":
        df["grp"] = rng.integers(0, n // 6, n)
        z = z + pd.Series(df.grp).map(dict(enumerate(rng.normal(0, 0.8, n // 6)))).to_numpy()
    df["Survived"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    folder.mkdir(parents=True, exist_ok=True)
    cut = int(n * 0.75)
    if kind == "group":
        g = df.grp.to_numpy()
        tr_mask = g < (n // 6) * 0.75
        train, te = df[tr_mask], df[~tr_mask].drop(columns="Survived")
    else:
        train, te = df.iloc[:cut], df.iloc[cut:].drop(columns="Survived")
    train.to_csv(folder / "train.csv", index=False)
    te.to_csv(folder / "test.csv", index=False)
    pd.DataFrame({"PassengerId": te.PassengerId, "Survived": 0}).to_csv(folder / "gender_submission.csv", index=False)
    return folder / "train.csv"
