import pandas as pd
import pytest

from common import training


@pytest.fixture(autouse=True)
def fixed_weight_params(monkeypatch):
    """Aritmetica legible: peso maximo 4, semivida de 100 dias."""
    monkeypatch.setattr(training, "MAX_SAMPLE_WEIGHT", 4)
    monkeypatch.setattr(training, "WEIGHT_HALF_LIFE_DAYS", 100)


def _df(ages_days: list[float]) -> pd.DataFrame:
    """Un DataFrame `ds`/`y` con una fila por edad (en dias).

    OJO: `_apply_recency_weights` calcula la antiguedad de cada fila relativa
    al MAXIMO `ds` de este mismo DataFrame (no contra un reloj externo), asi
    que para testear una edad especifica hay que incluir tambien una fila
    ancla en `age=0` -- si no, la unica fila pasada siempre es su propio
    maximo y su antiguedad da 0, sin importar que edad se le quiera simular.
    """
    now = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        {
            "ds": [now - pd.Timedelta(days=a) for a in ages_days],
            "y": [float(i) for i in range(len(ages_days))],
        }
    )


def _reps_for(out: pd.DataFrame, ds: pd.Timestamp) -> int:
    return int((out["ds"] == ds).sum())


def test_most_recent_row_gets_max_weight():
    out = training._apply_recency_weights(_df([0.0]))
    assert len(out) == training.MAX_SAMPLE_WEIGHT


def test_row_at_half_life_gets_half_weight():
    df = _df([0.0, training.WEIGHT_HALF_LIFE_DAYS])
    out = training._apply_recency_weights(df)
    expected = round(training.MAX_SAMPLE_WEIGHT * 0.5)
    assert _reps_for(out, df["ds"].iloc[1]) == expected


def test_very_old_row_never_drops_below_one_rep():
    df = _df([0.0, training.WEIGHT_HALF_LIFE_DAYS * 20])
    out = training._apply_recency_weights(df)
    assert _reps_for(out, df["ds"].iloc[1]) == 1


def test_output_length_matches_sum_of_expected_reps():
    ages = [0.0, 50.0, 100.0, 400.0]
    df = _df(ages)
    out = training._apply_recency_weights(df)

    expected_reps = [
        max(1, round(training.MAX_SAMPLE_WEIGHT * 0.5 ** (a / training.WEIGHT_HALF_LIFE_DAYS)))
        for a in ages
    ]
    assert len(out) == sum(expected_reps)


def test_repeated_rows_keep_their_original_values():
    df = _df([0.0, 200.0])
    out = training._apply_recency_weights(df)

    for ds, y in zip(df["ds"], df["y"]):
        matches = out[out["ds"] == ds]
        assert (matches["y"] == y).all()
