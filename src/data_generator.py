"""
Generates a synthetic grocery spoilage dataset.

We don't have a real dataset, so we create one ourselves using simple rules:
- higher temperature -> spoils faster
- higher humidity -> spoils faster
- longer delivery distance -> spoils faster
- better packaging -> spoils slower
- products with low initial shelf life -> spoil faster

Random noise is added so the data doesn't look perfectly clean.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

PRODUCTS = {
    "hardy": ["potato", "onion", "apple"],
    "normal": ["tomato", "carrot", "banana"],
    "fragile": ["milk", "berries", "leafy_vegetables"],
}

# shelf life range (in days) per category
SHELF_LIFE_RANGE = {
    "hardy": (15, 30),
    "normal": (5, 15),
    "fragile": (1, 5),
}

CATEGORY_LABEL = {"hardy": 0, "normal": 1, "fragile": 2}


def generate_data(n_rows=4000):
    rows = []

    for _ in range(n_rows):
        category = np.random.choice(["hardy", "normal", "fragile"])
        product = np.random.choice(PRODUCTS[category])

        temperature = np.random.uniform(5, 40)
        humidity = np.random.uniform(30, 95)
        distance_km = np.random.uniform(1, 300)
        packaging_quality = np.random.randint(1, 4)  # 1=poor, 2=medium, 3=good

        low, high = SHELF_LIFE_RANGE[category]
        initial_shelf_life = np.random.randint(low, high + 1)

        temp_effect = (temperature - 20) * 0.5
        humidity_effect = (humidity - 50) * 0.1
        distance_effect = distance_km * 0.04
        packaging_effect = packaging_quality * 1.5
        noise = np.random.normal(0, 1.5)

        spoilage_days = (
            initial_shelf_life
            - temp_effect
            - humidity_effect
            - distance_effect
            + packaging_effect
            + noise
        )
        spoilage_days = max(0.5, round(spoilage_days, 1))

        # relative threshold (30% of shelf life left) instead of a fixed number
        # of days, since "quick spoilage" means something different for a
        # potato (30 day shelf life) than for milk (3 day shelf life)
        spoilage_risk = 1 if spoilage_days <= 0.3 * initial_shelf_life else 0

        rows.append(
            {
                "temperature": round(temperature, 1),
                "humidity": round(humidity, 1),
                "distance_km": round(distance_km, 1),
                "product_type": product,
                "category": category,
                "category_label": CATEGORY_LABEL[category],
                "initial_shelf_life": initial_shelf_life,
                "packaging_quality": packaging_quality,
                "spoilage_days": spoilage_days,
                "spoilage_risk": spoilage_risk,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_data(4000)
    df.to_csv("data/grocery_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/grocery_data.csv")
    print(df["category"].value_counts())
    print(df["spoilage_risk"].value_counts())
