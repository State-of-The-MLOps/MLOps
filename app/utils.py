import codecs
import pickle
import os
import yaml
import zipfile
import socketserver
import time
import glob
import shutil
import subprocess
import io

import tensorflow as tf

from app.database import engine
from logger import L
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def zip_model(model_path):
    model_buffer = io.BytesIO()
    
    basedir = os.path.basename(model_path)
    
    with zipfile.ZipFile(model_buffer, "w") as zf:
        for root, dirs, files in os.walk(model_path):
            make_arcname = lambda x: os.path.join(root.split(basedir)[-1], x)
            for dr in dirs:
                dir_path = os.path.join(root, dr)
                zf.write(filename = dir_path, arcname = make_arcname(dr))
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(filename = file_path, arcname = make_arcname(file))
    
    return model_buffer

def get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    return free_port

def check_expr_over(experiment_id, experiment_name, experiment_path):
    minute = 60

    while True:
        time.sleep(1*minute)

        # 실험이 끝났는지 확인
        expr_list = subprocess.getoutput("nnictl experiment list")
        
        running_expr = [expr for expr in expr_list.split('\n') if experiment_id in expr]
        print(running_expr)
        if running_expr and ("DONE" in running_expr[0]):
            stop_expr = subprocess.getoutput("nnictl stop {}".format(
                                             experiment_id))
            L.info(stop_expr)
            break # 실험이 끝나면 무한루프 종료

        elif experiment_id not in expr_list:
            L.info(expr_list)
            break # 갑자기 누군가가 nnictl stop으로 다 꺼버렸을 상황에 대비

        else:
            model_path = os.path.join(experiment_path,
                                      "temp",
                                      "*_{}*".format(experiment_name))
            exprs = glob.glob(model_path)
            if len(exprs) > 3: # 모델파일이 너무 많아지지 않게 3개 넘으면 삭제
                exprs.sort()
                [shutil.rmtree(_) for _ in exprs[3:]]
    
    # 모델저장
    model_path = os.path.join(experiment_path,
                                "temp",
                                "*_{}*".format(experiment_name))
    exprs = glob.glob(model_path)
    if not exprs: # 모델파일이 하나도 없을 경우 그냥 종료
        return 0
    
    exprs.sort()
    exprs = exprs[0]
    metrics = os.path.basename(exprs).split("_")[:2] # metric 개수
    metrics = [float(metric) for metric in metrics]
    
    score_sql = """SELECT mae 
                   FROM atmos_model_metadata
                   ORDER BY mae;"""
    saved_score = engine.execute(score_sql).fetchone()

    if not saved_score or (metrics[0] < saved_score[0]):
        winner_model = os.path.join(os.path.join(experiment_path,
                                "temp",
                                experiment_name))
        os.rename(exprs, winner_model)
        m_buffer = zip_model(winner_model)
        encode_model = codecs.encode(pickle.dumps(m_buffer), "base64").decode()
        sql_save_model = "INSERT INTO model_core VALUES ('%s', '%s')"
        engine.execute(sql_save_model%(experiment_name, 
                                       encode_model))

        sql_save_score = "INSERT INTO atmos_model_metadata VALUES ('%s', '%s', '%s', '%s')"
        engine.execute(sql_save_score%(experiment_name, 
                                       experiment_id, 
                                       metrics[0], 
                                       metrics[1]))
        L.info("saved model %s %s"%(experiment_id, experiment_name))


