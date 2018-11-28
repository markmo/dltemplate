import yaml


def load_hyperparams(file_path):
    with open(file_path) as f:
        return yaml.load(f)


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if value is not None:
            merged[key] = value

    return merged
