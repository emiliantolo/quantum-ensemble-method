from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.models import BackendProperties, BackendConfiguration
from qiskit_aer.noise import NoiseModel
import time
from datetime import datetime
import json
import os
import importlib

class SimulatorBackend():

    def __init__(self):
        self.config_path = 'simulation/configs/'

    def download_backend_to_config(self, name='ibm_brisbane'):
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)
        service = QiskitRuntimeService(channel="ibm_quantum")
        real_backend = service.get_backend(name)
        conf = {
            'name': name,
            'date_created': datetime.now(),
            'properties': real_backend.properties().to_dict(),
            'configuration': real_backend.configuration().to_dict(),
            'dt': real_backend.dt
            }
        file_name = 'config-{}-{}.json'.format(name, time.time())
        json.dump(conf, open('{}/{}'.format(file_name), 'w'), default=str, indent=4)
        return file_name

    def gen_simulator_from_config(self, file_name=None):
        if file_name == None:
            # get last
            if not os.path.exists(self.config_path):
                raise ValueError('no configs found')
            l = os.listdir(self.config_path)
            if len(l) == 0:
                raise ValueError('no configs found')
            file_name = max(l, key=lambda c:float(c.split('-')[-1].split('.json')[0]))
        config = json.load(open('{}{}'.format(self.config_path, file_name)))
        configuration = BackendConfiguration.from_dict(config.get('configuration'))
        properties = BackendProperties.from_dict(config.get('properties'))
        dt = config.get('dt')
        name = dt = config.get('name')
        date_created = config.get('date_created')
        configuration.backend_name = "aer_simulator_{}_{}".format(name, date_created)
        noise_model = NoiseModel.from_backend_properties(properties, dt=dt)
        return AerSimulator(configuration=configuration, properties=properties, noise_model=noise_model)
    
    def gen_simulator_fake(self, name=None):
        #from qiskit.providers.fake_provider import FakeMumbaiV2
        #device_backend = FakeMumbaiV2()
        #return FakeMumbaiV2()
        if name == None:
            name = 'FakeMumbaiV2'
        module = importlib.import_module('qiskit.providers.fake_provider')
        sim_class = getattr(module, name)
        return sim_class()
