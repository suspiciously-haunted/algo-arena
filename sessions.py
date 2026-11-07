# sessions.py: Hudson River Trading -- Synthetic Market Prediction Challenge
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#####################################################
#
#   You should not need to read any of this code.
#   Go read algoarena.ipynb !
#
#####################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pandas as pd
from rich import print
from tqdm.notebook import tqdm

tqdm.pandas()

HALFWAY_BAR = 49
"""You trade at the close of this bar."""

HORIZON_BARS = 10
"""You hold your position for this many bars: until the close of bar 59."""

_DATA_DIR = Path(__file__).resolve().parent


@attrs.define(kw_only=True)
class BacktestFailure:
    sample: pd.Series
    """The test sample that your prediction errored on."""

    return_value: Any
    """The value that your prediction returned."""

    message: str
    """A more detailed description of the error."""


class Error(RuntimeError):
    def __init__(
        self,
        exc: Exception,
        sample: pd.Series,
        return_value: Any,
        message: str,
    ) -> None:
        super().__init__(message)
        self.exc = exc
        self.failure = BacktestFailure(sample=sample, return_value=return_value, message=message)


def _cov(x: pd.Series, y: pd.Series) -> float:
    return float(((x - x.mean()) * (y - y.mean())).mean())


def _sanitize(func: Callable[[pd.Series], float]) -> Callable[[pd.Series], float]:
    @functools.wraps(func)
    def wrapper(series: pd.Series) -> float:
        try:
            val = func(series)
        except Exception as e:
            raise Error(
                exc=e,
                sample=series,
                return_value=None,
                message="Your function raised an exception.",
            ) from None
        try:
            val = float(val)
        except (TypeError, ValueError) as e:
            raise Error(
                exc=e,
                sample=series,
                return_value=val,
                message="The return value was not a real number.",
            ) from None
        return val if math.isfinite(val) else 0

    return wrapper


@attrs.define
class _ErrorManager:
    last_error: Error | None = None

    @contextmanager
    def save_error(self) -> Generator[None, None, None]:
        try:
            yield
        except Error as e:
            self.last_error = e
            print(
                "Error! " + e.failure.message + "\n"
                "Run `sessions.get_last_error().sample` to see the sample you errored on."
            )
            raise e.exc from None

    def get_last_error(self) -> BacktestFailure | None:
        return self.last_error.failure if self.last_error else None


@functools.cache
def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(_DATA_DIR / name)


def _session_view(bars: pd.DataFrame, headlines: pd.DataFrame) -> pd.DataFrame:
    """One row per session: the first half's bars as arrays and its headlines as lists."""
    bars = bars[bars["bar_ix"] <= HALFWAY_BAR]
    headlines = headlines[headlines["bar_ix"] <= HALFWAY_BAR]
    sessions = pd.Index(np.sort(bars["session"].unique()), name="session")
    view = pd.DataFrame(index=sessions)
    for column in ("open", "high", "low", "close"):
        wide = bars.pivot(index="session", columns="bar_ix", values=column).reindex(sessions)
        view[column] = list(wide.to_numpy())
    grouped = headlines.groupby("session")
    for column, source in (("headlines", "headline"), ("headline_bars", "bar_ix")):
        lists = grouped[source].agg(list).reindex(sessions)
        view[column] = [value if isinstance(value, list) else [] for value in lists]
    return view


@attrs.define(kw_only=True)
class PredictionTask:
    bars_file: str
    headlines_file: str
    returns_file: str | None
    error_manager: _ErrorManager | None
    label: str

    def get_data(self) -> pd.DataFrame:
        df = _session_view(_read(self.bars_file), _read(self.headlines_file))
        if self.returns_file is not None:
            df = df.join(_read(self.returns_file).set_index("session")["returns"])
        return df

    def get_predictions(
        self,
        func: Callable[[pd.Series], float],
        *,
        subsample_fraction: float | None = None,
        subsample_seed: int = 1,
    ) -> pd.DataFrame:
        df = self.get_data()
        if subsample_fraction is not None:
            if subsample_fraction < 0.0 or subsample_fraction > 1.0:
                raise ValueError(f"{subsample_fraction=} must be between 0 and 1")
            df = df.sample(frac=subsample_fraction, random_state=subsample_seed).sort_index()
        # A separate copy, so a function that mutates its row's arrays can't change df.
        inputs = self.get_data().loc[df.index].drop(columns=["returns"], errors="ignore")
        with (self.error_manager or _ErrorManager()).save_error():
            return df.assign(pred=inputs.progress_apply(_sanitize(func), axis=1))

    @staticmethod
    def evaluate_predictions(
        df: pd.DataFrame,
        *,
        label: str,
    ) -> pd.Series:
        if "returns" not in df.columns:
            raise RuntimeError(f"you don't have ground truth labels for {label}")

        pnl = df["pred"] * df["returns"]
        portfolio = df["pred"].abs()
        result = dict(
            name=label,
            sharpe=pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252),
            total_pnl=pnl.sum(),
            frac_nonzero_predictions=(df["pred"] != 0).mean(),
            correlation=(
                _cov(df["pred"], df["returns"])
                / np.sqrt(_cov(df["pred"], df["pred"]) * _cov(df["returns"], df["returns"]) + 1e-12)
            ),
            mean_return_positive=df.query("pred > 0")["returns"].mean(),
            mean_return_negative=df.query("pred < 0")["returns"].mean(),
            mean_daily_portfolio_size=portfolio.mean(),
            annual_return=pnl.sum() / (portfolio.sum() + 1e-12),
        )

        ax = pnl.cumsum().reset_index(drop=True).plot(label=label)
        ax.set_title(f"Backtest result: {label}")
        ax.set_xlabel("Sessions")
        ax.set_ylabel("Cumulative PnL (units do not matter)")
        ax.set_yticklabels([])
        return pd.Series(result)

    def run_backtest(
        self,
        func: Callable[[pd.Series], float],
        *,
        subsample_fraction: float | None = None,
        subsample_seed: int = 1,
    ) -> pd.Series:
        df = self.get_predictions(
            func,
            subsample_fraction=subsample_fraction,
            subsample_seed=subsample_seed,
        )
        return PredictionTask.evaluate_predictions(df, label=f"{func.__name__} ({self.label})")


_error_manager = _ErrorManager()
_in_sample = PredictionTask(
    bars_file="ins_bars.parquet",
    headlines_file="ins_headlines.parquet",
    returns_file="ins_returns.parquet",
    error_manager=_error_manager,
    label="in-sample",
)
_out_of_sample = PredictionTask(
    bars_file="oos_bars.parquet",
    headlines_file="oos_headlines.parquet",
    returns_file=None,
    error_manager=_error_manager,
    label="out-of-sample",
)


def get_ins_full_bars() -> pd.DataFrame:
    """All 100 bars of every in-sample session, one row per bar.

    Out-of-sample you only see bars 0-49, so use this for research, not in your prediction.
    """
    return _read("ins_bars.parquet").copy()


def get_ins_full_headlines() -> pd.DataFrame:
    """Every headline of every in-sample session, including those after bar 49.

    Out-of-sample you only see headlines up to bar 49, so use this for research, not in your
    prediction.
    """
    return _read("ins_headlines.parquet").copy()


get_last_error = _error_manager.get_last_error
get_ins_data = _in_sample.get_data
get_oos_data = _out_of_sample.get_data
get_ins_predictions = _in_sample.get_predictions
get_oos_predictions = _out_of_sample.get_predictions
run_ins_backtest = _in_sample.run_backtest
run_oos_backtest = _out_of_sample.run_backtest


def write_submission(
    *,
    prediction_df: pd.DataFrame | None = None,
    prediction_func: Callable[[pd.Series], float] | None = None,
    path: str | Path = "predictions.csv",
) -> Path:
    """Write out-of-sample predictions to a CSV for upload to arcade.

    Pass either `prediction_df` (e.g. from `sessions.get_oos_predictions(...)`) or
    `prediction_func`, in which case the out-of-sample predictions are computed here. Arcade
    calls the id column `headline_id`; it holds the session number.
    """
    if prediction_df is None:
        if prediction_func is None:
            raise ValueError("Pass prediction_df=... or prediction_func=...")
        prediction_df = get_oos_predictions(prediction_func)

    if "session" not in prediction_df.columns:
        prediction_df = prediction_df.reset_index()
    for column in ("session", "pred"):
        if column not in prediction_df.columns:
            raise RuntimeError(f"Your predictions need a {column!r} column.")

    expected_sessions = get_oos_data().index
    if prediction_df["session"].duplicated().any():
        raise RuntimeError("Your predictions have duplicate session values.")
    expected = set(expected_sessions)
    got = set(prediction_df["session"])
    if got != expected:
        raise RuntimeError(
            "Your predictions have different rows to the out-of-sample dataset: "
            f"{len(expected - got)} sessions missing, {len(got - expected)} unexpected. "
            "Use sessions.get_oos_predictions(...) to build them."
        )

    try:
        pred = prediction_df["pred"].astype(np.float64)
    except (TypeError, ValueError) as e:
        raise RuntimeError("Your predictions are not all real numbers.") from e
    submission = (
        pd.DataFrame(
            {
                "headline_id": prediction_df["session"].astype(np.int64).to_numpy(),
                "pred": pred.where(np.isfinite(pred), 0.0).to_numpy(),
            }
        )
        .set_index("headline_id")
        .loc[expected_sessions.to_numpy()]
        .reset_index()
    )

    path = Path(path)
    submission.to_csv(path, index=False, float_format=None, lineterminator="\n")
    print(
        f"Wrote {len(submission)} predictions to [bold]{path.resolve()}[/bold].\n"
        "Upload this file on the challenge page in arcade to score it out-of-sample. "
        "Uploads are limited (see the challenge page), so make them count!"
    )
    return path


__all__ = [
    "HALFWAY_BAR",
    "HORIZON_BARS",
    "BacktestFailure",
    "Error",
    "get_ins_data",
    "get_ins_full_bars",
    "get_ins_full_headlines",
    "get_ins_predictions",
    "get_last_error",
    "get_oos_data",
    "get_oos_predictions",
    "run_ins_backtest",
    "run_oos_backtest",
    "write_submission",
]
