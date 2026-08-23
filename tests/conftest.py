"""Deja `common` (shared/common) y los DAGs (airflow/dags) importables,
replicando el mismo layout que docker-compose.yml monta en runtime.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent # tests/ -> raíz del repo
SHARED = ROOT / "shared"
DAGS = ROOT / "airflow" / "dags"

for path in (SHARED, DAGS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


