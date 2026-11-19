from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.bootstrap import bootstrap_database
from app.data.repositories.daily_metrics import (
    query_avg_order_value,
    query_gmv,
    query_order_count,
)


def main() -> None:
    dataset_dir = bootstrap_database()
    print(f"Loaded dataset from: {dataset_dir}")

    sample_date = "2018-08-10"
    print(query_gmv(sample_date))
    print(query_order_count(sample_date))
    print(query_avg_order_value(sample_date))


if __name__ == "__main__":
    main()
