from src.models.model import ModelFactory

if __name__ == "__main__":
    random_forest = ModelFactory.create_model("random_forest")
    print(f"RandomForest Model: {random_forest}")
