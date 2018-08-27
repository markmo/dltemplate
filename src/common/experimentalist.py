from datetime import datetime
import requests

EMS_URL = 'http://localhost:5000/'  # Experiment Management System URL


def make_experiment_name():
    return 'exp' + datetime.now().strftime('%Y%m%d%H%M%S')


def record_experiment(data):
    url = EMS_URL + 'experiments'
    response = requests.post(url, data=data)
    return response


def set_experiment_defaults(constants, defaults):
    for k, v in defaults.items():
        if k not in constants:
            constants[k] = v
