"""
Simple, manual counterfactual explanations.

We do NOT use SHAP / LIME / DiCE here. The idea is simpler: take the
current input, slightly change one feature at a time (temperature,
humidity, distance), re-run the prediction, and see if the risk flips
from High to Low. The first change that flips the prediction is
returned as the explanation.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "distance_km",
    "initial_shelf_life",
    "packaging_quality",
]

SEARCH_SETTINGS = {
    "temperature": {"step": -1, "max_steps": 20, "unit": "°C"},
    "humidity": {"step": -5, "max_steps": 12, "unit": "%"},
    "distance_km": {"step": -10, "max_steps": 15, "unit": "km"},
}


def predict_risk(model, input_dict):
    """Runs the model on a single input row and returns 0 (Low) or 1 (High)."""
    row = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    return int(model.predict(row)[0])


def find_counterfactual(input_dict, model):
    """
    Tries small changes to temperature, humidity and distance_km to see if
    the spoilage risk prediction can be flipped from High (1) to Low (0).

    Returns a plain-English sentence describing the first change that works,
    or None if no small change helped.
    """
    current_risk = predict_risk(model, input_dict)

    if current_risk == 0:
        return None

    for feature, settings in SEARCH_SETTINGS.items():
        original_value = input_dict[feature]

        for step_num in range(1, settings["max_steps"] + 1):
            new_value = original_value + settings["step"] * step_num
            if new_value <= 0:
                break

            test_input = dict(input_dict)
            test_input[feature] = new_value

            new_risk = predict_risk(model, test_input)

            if new_risk == 0:
                unit = settings["unit"]
                return (
                    f"If {feature.replace('_', ' ')} is reduced from "
                    f"{original_value}{unit} to {new_value}{unit}, "
                    f"the predicted spoilage risk decreases to Low."
                )

    return None
