from src.data.data_preprocessor import DataProcessor
from src.data.data_loader import DataLoader

if __name__ == "__main__":
    loader = DataLoader()
    processor = DataProcessor()

    df = loader.load_data()
    X, y = processor.fit_transform(df)
    print(f"Transformed features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
