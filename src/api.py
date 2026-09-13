
"""
FastAPI app for the grocery spoilage project.
"""

import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.counterfactual import find_counterfactual

app = FastAPI(title="Smart Grocery Spoilage Prediction")

CATEGORY_NAMES = {
    0: "hardy",
    1: "normal",
    2: "fragile",
}

PRODUCT_CATEGORY = {
    "potato": "hardy",
    "onion": "hardy",
    "apple": "hardy",
    "tomato": "normal",
    "carrot": "normal",
    "banana": "normal",
    "milk": "fragile",
    "berries": "fragile",
    "leafy_vegetables": "fragile",
}


def load_models():
    models = {}

    with open("models/router.pkl", "rb") as f:
        models["router"] = pickle.load(f)

    with open("models/hardy_model.pkl", "rb") as f:
        models["hardy"] = pickle.load(f)

    with open("models/normal_model.pkl", "rb") as f:
        models["normal"] = pickle.load(f)

    with open("models/fragile_model.pkl", "rb") as f:
        models["fragile"] = pickle.load(f)

    return models


models = load_models()


class PredictRequest(BaseModel):
    temperature: float = Field(
        ...,
        description="Temperature in Celsius"
    )

    humidity: float = Field(
        ...,
        description="Humidity in percent"
    )

    distance_km: float = Field(
        ...,
        description="Delivery distance in km"
    )

    product_type: str = Field(
        ...,
        description="e.g. milk, potato, tomato"
    )

    initial_shelf_life: int = Field(
        ...,
        description="Shelf life in days"
    )

    packaging_quality: int = Field(
        2,
        ge=1,
        le=3,
        description="1=poor, 2=medium, 3=good (optional, default=2)"
    )


def find_causes(input_dict, risk):
    """
    Simple rule-based reasons for why the item is High risk.
    No model is involved in generating these reasons.
    """

    if risk == 0:
        return []

    causes = []

    if input_dict["temperature"] > 25:
        causes.append(
            f"Temperature is high ({input_dict['temperature']}°C)."
        )

    if input_dict["humidity"] > 70:
        causes.append(
            f"Humidity is high ({input_dict['humidity']}%)."
        )

    if input_dict["distance_km"] > 50:
        causes.append(
            f"Delivery distance is long ({input_dict['distance_km']} km)."
        )

    if input_dict["initial_shelf_life"] <= 5:
        causes.append(
            f"This product has a short shelf life "
            f"({input_dict['initial_shelf_life']} days)."
        )

    if input_dict["packaging_quality"] == 1:
        causes.append(
            "Packaging quality is poor."
        )

    if not causes:
        causes.append(
            "A combination of factors is pushing the risk higher."
        )

    return causes


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(request: PredictRequest):
    input_dict = request.dict()

    # ---------------------------------------------------------
    # Router model
    # ---------------------------------------------------------
    #
    # The router predicts the product category using ONLY:
    # temperature, humidity, distance and shelf life.
    #
    # It does NOT use the product name.
    #
    router_features = pd.DataFrame([
        {
            "temperature": input_dict["temperature"],
            "humidity": input_dict["humidity"],
            "distance_km": input_dict["distance_km"],
            "initial_shelf_life": input_dict["initial_shelf_life"],
        }
    ])

    router_label = int(
        models["router"].predict(router_features)[0]
    )

    category = CATEGORY_NAMES[router_label]

    # ---------------------------------------------------------
    # Specialist model
    # ---------------------------------------------------------

    specialist = models[category]

    specialist_features = pd.DataFrame([
        {
            "temperature": input_dict["temperature"],
            "humidity": input_dict["humidity"],
            "distance_km": input_dict["distance_km"],
            "initial_shelf_life": input_dict["initial_shelf_life"],
            "packaging_quality": input_dict["packaging_quality"],
        }
    ])

    risk = int(
        specialist.predict(specialist_features)[0]
    )

    probability = float(
        specialist.predict_proba(specialist_features)[0][1]
    )

    # ---------------------------------------------------------
    # Counterfactual explanation
    # ---------------------------------------------------------

    cf_input = {
        "temperature": input_dict["temperature"],
        "humidity": input_dict["humidity"],
        "distance_km": input_dict["distance_km"],
        "initial_shelf_life": input_dict["initial_shelf_life"],
        "packaging_quality": input_dict["packaging_quality"],
    }

    counterfactual = find_counterfactual(
        cf_input,
        specialist
    )

    # ---------------------------------------------------------
    # Risk causes
    # ---------------------------------------------------------

    causes = find_causes(
        input_dict,
        risk
    )

    # ---------------------------------------------------------
    # Product category vs router prediction
    # ---------------------------------------------------------
    #
    # The product has a usual category.
    #
    # Example:
    # potato -> hardy
    # milk   -> fragile
    #
    # But the router doesn't see product_type.
    # Therefore it can predict a different category based
    # only on environmental conditions.
    #

    expected_category = PRODUCT_CATEGORY.get(
        input_dict["product_type"]
    )

    router_note = None

    if (
        expected_category
        and expected_category != category
    ):
        router_note = (
            f"Note: "
            f"{input_dict['product_type'].replace('_', ' ')} "
            f"is usually '{expected_category}', "
            f"but the router predicted '{category}' "
            f"based purely on these conditions "
            f"(it doesn't see the product name)."
        )

    # ---------------------------------------------------------
    # API response
    # ---------------------------------------------------------

    return {
        "router_model": category,

        "expected_category": expected_category,

        "router_note": router_note,

        "current_temperature": input_dict[
            "temperature"
        ],

        "prediction": (
            "High Spoilage Risk"
            if risk == 1
            else "Low Spoilage Risk"
        ),

        "probability": round(
            probability,
            2
        ),

        "causes": causes,

        "counterfactual": (
            counterfactual
            if counterfactual
            else "No simple change found to reduce risk."
        ),
    }

