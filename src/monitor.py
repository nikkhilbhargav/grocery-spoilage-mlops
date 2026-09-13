"""
Very basic data drift check using Evidently.

The idea: over time, the kind of orders our API receives in production
might look different from the data we trained on (e.g. hotter weather in
summer, longer delivery distances). This script compares our original
training data ("reference") against a newer batch ("current") and
generates an HTML report showing whether the distributions have shifted.

This is NOT a real-time monitoring service, just a script you run manually
whenever you want to check for drift.
"""

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

COLUMNS_TO_CHECK = ["temperature", "humidity", "distance_km"]


def make_fake_current_data(reference_df, n_rows=500):
    """
    Simulates a new batch of production data with a slight shift
    (e.g. warmer temperatures) so the drift report has something to show.
    """
    current = reference_df.sample(n_rows, random_state=1).copy()
    current["temperature"] = current["temperature"] + np.random.normal(5, 2, n_rows)
    current["distance_km"] = current["distance_km"] * 1.3
    return current


def main():
    reference = pd.read_csv("data/grocery_data.csv")
    current = make_fake_current_data(reference)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[COLUMNS_TO_CHECK],
        current_data=current[COLUMNS_TO_CHECK],
    )
    report.save_html("drift_report.html")
    print("Drift report saved to drift_report.html")


if __name__ == "__main__":
    main()
