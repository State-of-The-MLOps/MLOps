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
    """
    predict API 호출을 받았을 때 사용될 ML 모델을 로드하는 클래스입니다.

    Attributes:
        model_name(str): 예측을 실행할 모델의 이름
        model(obj): 모델이 저장될 인스턴스 변수
        query(str): 입력받은 모델이름을 기반으로 데이터베이스에서 모델을 불러오는 SQL query입니다.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.query = """
                SELECT model_file
                FROM model_core
                WHERE model_name='{}';
            """.format(self.model_name)

    def load_model(self):
        """
        본 클래스를 상속받았을 때 이 함수를 구현하지 않으면 예외를 발생시킵니다.
        """
        raise Exception

    def predict_target(self, target_data):
        """
        데이터베이스에서 불러와 인스턴스 변수에 저장된 모델을 기반으로 예측을 수행합니다.

        Args:
            target_data: predict API 호출 시 입력받은 값입니다. 자료형은 모델에 따라 다릅니다.

        Returns:
            예측된 값을 반환 합니다. 자료형은 모델에 따라 다릅니다.
        """
        return self.model.predict(target_data)


class ScikitLearnModel(CoreModel):
    """
    Scikit learn 라이브러리 기반의 모델을 불러오기 위한 클래스입니다.
    Examples:
        >>> sk_model = ScikitLearnModel("my_model")
        >>> sk_model.load_model()
        >>> sk_model.predict_target(target)
        predict result
    """
    def __init__(self, *args):
        super().__init__(*args)

    def load_model(self):
        """
        모델을 데이터베이스에서 불러와 인스턴스 변수에 저장하는 함수 입니다. 상속받은 부모클래스의 인스턴스 변수를 이용하며, 반환 값은 없습니다.
        """
        _model = engine.execute(self.query).fetchone()
        if _model is None:
            raise ValueError('Model Not Found!')

        self.model = pickle.loads(
            codecs.decode(_model[0], 'base64')
        )


class TensorFlowModel(CoreModel):
    """
    Tensorflow 라이브러리 기반의 모델을 불러오기 위한 클래스입니다.
    Examples:
        >>> tf_model = TensorflowModel("my_model")
        >>> tf_model.load_model()
        >>> tf_model.predict_target(target)
        predict result
    """
    def __init__(self, *args):
        super().__init__(*args)

    def load_model(self):
        """
        모델을 데이터베이스에서 불러와 인스턴스 변수에 저장하는 함수 입니다. 상속받은 부모클래스의 인스턴스 변수를 이용하며, 반환 값은 없습니다.
        """
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
    """
    NNI 실험을 시작하기 위한 config.yml파일을 작성하는 함수 입니다.

    Args:
        path(str): 실험의 경로
        experiment_name(str): 실험의 이름
        experimenter(str): 실험자의 이름
        model_name(str): 모델의 이름
        version(float): 버전

    Returns:
        반환 값은 없으며 입력받은 경로로 yml파일이 작성됩니다.
    """
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
    """
    입력받은 모델의 경로를 찾아가 모델을 압축하여 메모리 버퍼를 반환합니다.

    Args:
        model_path(str): 모델이 있는 경로입니다.

    Returns:
        memory buffer: 모델을 압축한 메모리 버퍼를 반환합니다.

    Note:
        모델을 보조기억장치에 파일로 저장하지 않습니다.
    """
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
    """
    호출 즉시 사용가능한 포트번호를 반환합니다.

    Returns:
        현재 사용가능한 포트번호

    Examples:
        >>> avail_port = get_free_port() # 사용 가능한 포트, 그때그때 다름
        >>> print(avail_port)
        45675
    """
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
    return free_port


def check_expr_over(experiment_id, experiment_name, experiment_path):
    """
    train API에서 사용되기 위하여 만들어 졌습니다. experiment_id를 입력받아 해당 id를 가진 nni 실험을 모니터링 합니다. 현재 추상화되어있지 않아 코드 재사용성이 부족하며 개선이 필요합니다.
    * 파일로 생성되는 모델이 너무 많아지지 않도록 유지합니다.(3개 이상 모델이 생성되면 성능순으로 3위 미만은 삭제)
    * nnictl experiment list(shell command)를 주기적으로 호출하여 실험이 현제 진행중인지 파악합니다.
    * 실험의 상태가 DONE으로 변경되면 최고점수 모델을 데이터베이스에 저장하고 nnictl stop experiment_id를 실행하여 실험을 종료한 후 프로세스가 종료됩니다.

    Args:
        experiment_id(str)
        experiment_name(str)
        experiment_path(str)
    """
    minute = 60

    while True:
        time.sleep(1*minute)

        expr_list = subprocess.getoutput("nnictl experiment list")
        
        running_expr = [expr for expr in expr_list.split('\n') if experiment_id in expr]
        print(running_expr)
        if running_expr and ("DONE" in running_expr[0]):
            stop_expr = subprocess.getoutput("nnictl stop {}".format(
                                             experiment_id))
            L.info(stop_expr)
            break

        elif experiment_id not in expr_list:
            L.info(expr_list)
            break

        else:
            model_path = os.path.join(experiment_path,
                                      "temp",
                                      "*_{}*".format(experiment_name))
            exprs = glob.glob(model_path)
            if len(exprs) > 3:
                exprs.sort()
                [shutil.rmtree(_) for _ in exprs[3:]]


    model_path = os.path.join(experiment_path,
                                "temp",
                                "*_{}*".format(experiment_name))
    exprs = glob.glob(model_path)
    if not exprs:
        return 0
    
    exprs.sort()
    exprs = exprs[0]
    metrics = os.path.basename(exprs).split("_")[:2]
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


