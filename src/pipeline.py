"""
A very simple Prefect pipeline that ties together the steps we'd otherwise
run by hand:

    generate data -> train models -> evaluate -> save models

Prefect just gives us task tracking, logs and automatic retries. We add
retries=2 on the data generation step to show how Prefect can re-run a
task if something goes wrong (e.g. a temporary file write error).
"""

from prefect import flow, task

from src import data_generator, train


@task(retries=2, retry_delay_seconds=5)
def generate_data_task():
    df = data_generator.generate_data(4000)
    df.to_csv("data/grocery_data.csv", index=False)
    print(f"Generated {len(df)} rows")
    return df


@task
def train_models_task(df):
    """Trains router + all 3 specialist models and saves them to models/."""
    import mlflow

    mlflow.set_experiment("grocery-spoilage")

    router = train.train_router(df)
    train.save_model(router, "models/router.pkl")

    hardy_model = train.train_specialist(df, "hardy", "logistic_regression")
    train.save_model(hardy_model, "models/hardy_model.pkl")

    normal_model = train.train_specialist(df, "normal", "random_forest")
    train.save_model(normal_model, "models/normal_model.pkl")

    fragile_model = train.train_specialist(df, "fragile", "xgboost")
    train.save_model(fragile_model, "models/fragile_model.pkl")

    print("All 4 models trained and saved.")


@flow(name="grocery-spoilage-training-pipeline")
def training_pipeline():
    df = generate_data_task()
    train_models_task(df)


if __name__ == "__main__":
    training_pipeline()
