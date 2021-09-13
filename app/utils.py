import codecs
import pickle
import os
import yaml
import zipfile

import tensorflow as tf

from app.database import engine


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class CoreModel:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.query = """
                SELECT model_file
                FROM model_core
                WHERE model_name='{}';
            """.format(self.model_name)

    def load_model(self):
        raise Exception

    def predict_target(self, target_data):
        return self.model.predict(target_data)


class ScikitLearnModel(CoreModel):
    def __init__(self, *args):
        super().__init__(*args)

    def load_model(self):
        _model = engine.execute(self.query).fetchone()
        if _model is None:
            raise ValueError('Model Not Found!')

        self.model = pickle.loads(
            codecs.decode(_model[0], 'base64')
        )


class TensorFlowModel(CoreModel):
    def __init__(self, *args):
        super().__init__(*args)

    def load_model(self):
        _model = engine.execute(self.query).fetchone()
        if _model is None:
            raise ValueError('Model Not Found!')
        model_buffer = pickle.loads(codecs.decode(_model[0], "base64"))
        model_path = os.path.join(base_dir, "tf_model", self.model_name)

        with zipfile.ZipFile(model_buffer, "r") as bf:
            bf.extractall(model_path)
        self.model = tf.keras.models.load_model(model_path)


my_model = TensorFlowModel('test_model')
my_model.load_model()


def write_yml(
    path,
    experiment_name,
    experimenter,
    model_name,
    version
):
    with open('{}/{}.yml'.format(path, model_name), 'w') as yml_config_file:
        yaml.dump({
            'authorName': f'{experimenter}',
            'experimentName': f'{experiment_name}',
            'trialConcurrency': 1,
            'maxExecDuration': '1h',
            'maxTrialNum': 10,
            'trainingServicePlatform': 'local',
            'searchSpacePath': 'search_space.json',
            'useAnnotation': False,
            'tuner': {
                'builtinTunerName': 'Anneal',
                'classArgs': {
                    'optimize_mode': 'minimize'
                }},
            'trial': {
                'command': 'python trial.py -e {} -n {} -m {} -v {}'.format(
                    experimenter,
                    experiment_name,
                    model_name,
                    version
                ),
                'codeDir': '.'
            }}, yml_config_file, default_flow_style=False)

        yml_config_file.close()

    return
