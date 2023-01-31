from datasets import load_dataset, load_from_disk

dataset = load_dataset("quantumiracle-git/robotinder-data")
dataset.save_to_disk("test.hf")
print(dataset, dataset['train'][0])