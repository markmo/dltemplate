from datetime import datetime
import requests
from requests.exceptions import ConnectionError

EMS_URL = 'http://localhost:5000/'  # Experiment Management System URL


def make_experiment_name():
    return 'exp' + datetime.now().strftime('%Y%m%d%H%M%S')


def record_experiment(data):
    url = EMS_URL + 'experiments'
    try:
        response = requests.post(url, data=data)
        return response
    except ConnectionError:
        print('Experiment Management System asleep!')
        return None


def set_experiment_defaults(constants, defaults):
    for k, v in defaults.items():
        if k not in constants:
            constants[k] = v
