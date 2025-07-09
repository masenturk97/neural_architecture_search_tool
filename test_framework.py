from core.nn_search_engine import NNSearchEngine

params_dict = {
    "dataset": "datasets/sample_binary_classification_dataset.csv",
    "populationSize": 4,
    "maxGenerationCount": 2,
    "taskType": 2
}

search_engine = NNSearchEngine(params_dict)

for iteration_models in search_engine:
    print(f"Generation: {search_engine.generationCount}")
    for model in iteration_models:
        print(model.toDict())

print(f"Best Solution: {search_engine.finalBestFoundModel().toDict()}")