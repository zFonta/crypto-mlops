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

# ---------------------------------------------------------------------------
# Complementos: parametros reales de produccion e invariantes generales
# ---------------------------------------------------------------------------

def test_production_params_cutoffs():
    """Con los valores reales (MAX=3, semivida=120) fija los cortes que documenta
    el README, en vez de la aritmetica 'legible' 4/100 del resto del archivo.

    Ojo el redondeo bancario: a 120 dias el peso crudo es 1.5 y round(1.5)=2
    (redondeo a par), asi que la fila de la semivida cae del lado del 2, no del 1.
    """
    # NO usa el fixture fixed_weight_params: monkeypatchea a los valores reales.
    saved = (training.MAX_SAMPLE_WEIGHT, training.WEIGHT_HALF_LIFE_DAYS)
    training.MAX_SAMPLE_WEIGHT, training.WEIGHT_HALF_LIFE_DAYS = 3, 120
    try:
        df = _df([0.0, 31.0, 33.0, 119.0, 121.0, 364.0])
        out = training._apply_recency_weights(df)
        reps = {int(a): _reps_for(out, df["ds"].iloc[i]) for i, a in
                enumerate([0, 31, 33, 119, 121, 364])}
    finally:
        training.MAX_SAMPLE_WEIGHT, training.WEIGHT_HALF_LIFE_DAYS = saved

    assert reps[0] == 3      # la hora mas reciente pesa el maximo
    assert reps[31] == 3     # ~antes del primer corte
    assert reps[33] == 2     # ~despues del primer corte
    assert reps[119] == 2    # antes de la semivida
    assert reps[121] == 1    # despues de la semivida
    assert reps[364] == 1    # el borde de la ventana de entrenamiento


def test_weight_is_monotonic_non_increasing():
    """Mas viejo nunca pesa mas que mas nuevo."""
    df = _df([0.0, 10.0, 50.0, 100.0, 300.0, 500.0])
    out = training._apply_recency_weights(df)
    reps = [_reps_for(out, ds) for ds in sorted(df["ds"], reverse=True)]  # de mas nuevo a mas viejo
    assert all(a >= b for a, b in zip(reps, reps[1:])), reps


def test_no_row_exceeds_max_weight():
    """Ningun peso supera MAX_SAMPLE_WEIGHT (la fila mas reciente marca el techo)."""
    df = _df([0.0, 5.0, 40.0, 200.0])
    out = training._apply_recency_weights(df)
    max_reps = max(_reps_for(out, ds) for ds in df["ds"])
    assert max_reps <= training.MAX_SAMPLE_WEIGHT


def test_no_hour_is_lost():
    """Repetir filas nunca debe PERDER una hora: el set de ds unicos del output
    tiene que ser igual al del input (si el modelo deja de ver una parte del
    anio, el trend se estima peor y nada en el dashboard lo delata)."""
    df = _df([0.0, 10.0, 100.0, 250.0, 400.0])
    out = training._apply_recency_weights(df)
    assert set(out["ds"]) == set(df["ds"])


def test_does_not_mutate_input():
    """La expansion no debe alterar el DataFrame de entrada."""
    df = _df([0.0, 100.0, 300.0])
    before = len(df)
    training._apply_recency_weights(df)
    assert len(df) == before
