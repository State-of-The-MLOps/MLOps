import codecs
import pickle
import os
import yaml
import zipfile

import tensorflow as tf

from app.database import engine


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class MyModel:
    def __init__(self):
        self._my_model = None

    def load_tf_model(self, model_name):
        """
        * DB에 있는 텐서플로우 모델을 불러옵니다. 
        * 모델은 zip형식으로 압축되어 binary로 저장되어 있습니다.
        * 모델의 이름을 받아 압축 해제 및 tf_model폴더 아래에 저장한 후 로드하여 
          텐서플로우 모델 객체를 반환합니다.
        """

        query = f"""SELECT model_file
                    FROM model_core
                    WHERE model_name='{model_name}';"""

        bin_data = engine.execute(query).fetchone()[0]

        model_buffer = pickle.loads(codecs.decode(bin_data, "base64"))
        model_path = os.path.join(base_dir, "tf_model", model_name)

        with zipfile.ZipFile(model_buffer, "r") as bf:
            bf.extractall(model_path)
        tf_model = tf.keras.models.load_model(model_path)

        return tf_model

    def load_model(self):
        self._my_model = self.load_tf_model('test_model')

    @property
    def my_model(self):
        return self._my_model


my_model = MyModel()
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
