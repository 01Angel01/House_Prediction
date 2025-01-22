from mlflow.tracking import MlflowClient

client = MlflowClient()

# Coba ambil experiment
experiment = client.get_experiment_by_name("house_price_experiment")

if experiment:
    print("Experiment ditemukan:", experiment.experiment_id)
else:
    print("Experiment tidak ditemukan!")
