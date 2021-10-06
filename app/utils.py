import codecs
import glob
import io
import multiprocessing
import os
import pickle
import re
import shutil
import socketserver
import subprocess
import time
import zipfile


import tensorflow as tf
import yaml

from app.database import engine
from app.query import *
from logger import L

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

physical_devices = tf.config.list_physical_devices("GPU")
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
            """.format(
            self.model_name
        )

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
            raise ValueError("Model Not Found!")

        self.model = pickle.loads(codecs.decode(_model[0], "base64"))


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
            raise ValueError("Model Not Found!")
        model_buffer = pickle.loads(codecs.decode(_model[0], "base64"))
        model_path = os.path.join(base_dir, "tf_model", self.model_name)

        with zipfile.ZipFile(model_buffer, "r") as bf:
            bf.extractall(model_path)
        self.model = tf.keras.models.load_model(model_path)


my_model = TensorFlowModel("test_model")
my_model.load_model()


def write_yml(path, experiment_name, experimenter, model_name, version):
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
    with open("{}/{}.yml".format(path, model_name), "w") as yml_config_file:
        yaml.dump(
            {
                "authorName": f"{experimenter}",
                "experimentName": f"{experiment_name}",
                "trialConcurrency": 1,
                "maxExecDuration": "1h",
                "maxTrialNum": 10,
                "trainingServicePlatform": "local",
                "searchSpacePath": "search_space.json",
                "useAnnotation": False,
                "tuner": {
                    "builtinTunerName": "Anneal",
                    "classArgs": {"optimize_mode": "minimize"},
                },
                "trial": {
                    "command": "python trial.py -e {} -n {} -m {} -v {}".format(
                        experimenter, experiment_name, model_name, version
                    ),
                    "codeDir": ".",
                },
            },
            yml_config_file,
            default_flow_style=False,
        )

        yml_config_file.close()

    return


class NniWatcher:
    """
    experiment_id를 입력받아 해당 id를 가진 nni 실험을 모니터링하고 모델 파일을 관리해주는 클래스입니다.
    생성되는 scikit learn 모델을 DB의 임시 테이블에 저장하여 주기적으로 업데이트 합니다.
    이후 실험의 모든 프로세스가 종료되면 가장 성능이 좋은 모델과 점수를 업데이트 합니다.
    
    Attributes:
        experiment_id(str): nni experiment를 실행할 때 생성되는 id
        experiment_name(str): 실험의 이름
        experimenter(str): 실험자의 이름
        version(str): 실험의 버전
        minute(int): 감시 주기
        is_kill(bool, default=True)
        is_update(bool, default=True)
        top_cnt(int, default=3)
        evaluation_criteria(str, default="val_mae")

    Examples:
        >>> watcher = NniWatcher(expr_id, experiment_name, experimenter, version)
        >>> watcher.execute()
    """
    def __init__(
        self,
        experiment_id,
        experiment_name,
        experimenter,
        version,
        minute=1,
        is_kill=True,
        is_update=True,
        top_cnt=3,
        evaluation_criteria="val_mae",
    ):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.experimenter = experimenter
        self.version = version
        self.is_kill = is_kill
        self.is_update = is_update
        self.top_cnt = top_cnt
        self.evaluation_criteria = evaluation_criteria
        self._wait_minute = minute * 20
        self._experiment_list = None
        self._running_experiment = None

    def excute(self):
        """
        모든 함수를 실행합니다.
        """
        self.watch_process()
        self.model_final_update()

    def get_running_experiment(self):
        """
        실행중인 실험의 목록을 가져와 저장합니다.
        """
        self._experiment_list = subprocess.getoutput("nnictl experiment list")
        self._running_experiment = [
            expr
            for expr in self._experiment_list.split("\n")
            if self.experiment_id in expr
        ]
        L.info(self._running_experiment)

    def watch_process(self):
        """
        사용자가 지정한 시간을 주기로 실험 프로세스가 진행 중인지 감시하고 "DONE"상태로 변경되면 실험을 종료합니다.
        모델의 score를 DB에 주기적으로 업데이트 해줍니다.
        """
        if self.is_kill:
            while True:
                self.get_running_experiment()
                if self._running_experiment and ("DONE" in self._running_experiment[0]):
                    _stop_expr = subprocess.getoutput(
                        "nnictl stop {}".format(self.experiment_id)
                    )
                    L.info(_stop_expr)
                    break

                elif self.experiment_id not in self._experiment_list:
                    L.error("Experiment ID not in Current Experiment List")
                    L.info(self._experiment_list)
                    break

                else:
                    if self.is_update:
                        self.model_keep_update()
                time.sleep(self._wait_minute)

    def model_keep_update(self):
        """
        scikit learn 모델의 성능을 DB에 업데이트 합니다.
        """
        engine.execute(
            UPDATE_TEMP_MODEL_DATA.format(
                self.experiment_name, self.evaluation_criteria, self.top_cnt
            )
        )

    def model_final_update(self):
        """
        실험 종료시 실행되는 함수로 모델의 최종 점수와 모델 파일을 DB에 업데이트 해줍니다.
        """
        final_result = engine.execute(
            SELECT_TEMP_MODEL_BY_EXPR_NAME.format(
                self.experiment_name, self.evaluation_criteria
            )
        ).fetchone()

        saved_result = engine.execute(
            SELECT_MODEL_METADATA_BY_EXPR_NAME.format(self.experiment_name)
        ).fetchone()

        a = pickle.loads(codecs.decode(final_result.model_file, "base64"))
        pickled_model = codecs.encode(pickle.dumps(a), "base64").decode()

        if saved_result is None:
            engine.execute(
                INSERT_MODEL_CORE.format(final_result.model_name, pickled_model)
            )
            engine.execute(
                INSERT_MODEL_METADATA.format(
                    self.experiment_name,
                    final_result.model_name,
                    self.experimenter,
                    self.version,
                    final_result.train_mae,
                    final_result.val_mae,
                    final_result.train_mse,
                    final_result.val_mse,
                )
            )
        elif (
            saved_result[self.evaluation_criteria]
            > final_result[self.evaluation_criteria]
        ):
            engine.execute(
                UPDATE_MODEL_CORE.format(pickled_model, saved_result.model_name)
            )
            engine.execute(
                UPDATE_MODEL_METADATA.format(
                    final_result.train_mae,
                    final_result.val_mae,
                    final_result.train_mse,
                    final_result.val_mse,
                    self.experiment_name,
                )
            )

        engine.execute(DELETE_ALL_EXPERIMENTS_BY_EXPR_NAME.format(self.experiment_name))


def zip_model(model_path):
    """
    입력받은 모델의 경로를 찾아가 모델을 압축하여 메모리 버퍼를 반환합니다.

    Args:
        model_path(str): 모델이 있는 경로입니다.

    Returns:
        memory buffer: 모델을 압축한 메모리 버퍼를 반환합니다.

    Note:
        모델을 디스크에 파일로 저장하지 않습니다.
    """
    model_buffer = io.BytesIO()

    basedir = os.path.basename(model_path)

    with zipfile.ZipFile(model_buffer, "w") as zf:
        for root, dirs, files in os.walk(model_path):

            def make_arcname(x):
                return os.path.join(root.split(basedir)[-1], x)

            for dr in dirs:
                dir_path = os.path.join(root, dr)
                zf.write(filename=dir_path, arcname=make_arcname(dr))
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(filename=file_path, arcname=make_arcname(file))

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


class ExperimentOwl:
    """
    experiment_id를 입력받아 해당 id를 가진 nni 실험을 모니터링하고 모델 파일을 관리해주는 클래스입니다.
    필요한 기능을 instance.add("method name") 메서드로 추가하여 사용할 수 있습니다.

    현재 보유한 기능
    1. (기본)nnictl experiment list(shell command)를 주기적으로 호출하여 실험이 현재 진행중인지 파악합니다.
       실험의 상태가 DONE으로 변경되면 최고점수 모델을 데이터베이스에 저장하고 nnictl stop experiment_id를 실행하여 실험을 종료한 후 프로세스가 종료됩니다.

    2. 파일로 생성되는 모델이 너무 많아지지 않도록 유지합니다.(3개 이상 모델이 생성되면 성능순으로 3위 미만은 삭제) instance 생성 시 
       mfile_manage = False로 기능을 사용하지 않을 수 있습니다.(default True)

    3. (method) update_tfmodelbd
       텐서플로우를 이용한 실험 시 생성되는 모델을 실험이 종료되면 DB에 저장하거나 점수가 향상되었을 시 업데이트 해줍니다.

    4. (method) modelfile_cleaner
       모든 실험이 종료되고 모델이 저장되면 temp 폴더에 있는 모델파일들을 모두 지워줍니다.

    Attributes:
        experiment_id(str): nni experiment를 실행할 때 생성되는 id
        experiment_name(str): 실험의 이름
        experiment_path(str): 실험코드가 있는 경로
        mfile_manage(bool, default=True): 주기적으로 파일 삭제 여부
        time(int or float, default=5): 감시주기(분)

    Examples:
        >>> owl = ExperimentOwl(id, name, path)
        >>> owl.add("update_tfmodeldb")
        >>> owl.add("modelfile_cleaner")
        >>> owl.execute()
    """

    def __init__(self, experiment_id, experiment_name, experiment_path, mfile_manage = True, time = 5):
        self.__minute = 60
        self.time = time * self.__minute
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path
        self.mfile_manage = mfile_manage
        self.__func_list = [self.main]


    def execute(self):
        """
        instance.add("method name")으로 저장된 메서드들을 순서대로 모두 실행시킵니다.
        """
        for func in self.__func_list:
            func()


    def add(self, func_name):
        func = getattr(self, func_name)
        self.__func_list.append(func)


    def main(self):
        """
        ExperimentOwl클래스로 인스턴스를 생성 후 실행시 기본적으로 실행되는 기능입니다.
        사용자가 지정한 시간을 주기로 실험 프로세스가 진행 중인지 감시하고 "DONE"상태로 변경되면 실험을 종료합니다.
        인스턴스 생성 시 mfile_manage옵션이 True이면 모델 파일이 너무 많아지지 않게 점수 순서로 3위 이하는 삭제합니다.(default True)
        """
        while True:
            time.sleep(self.__minute)

            expr_list = subprocess.getoutput("nnictl experiment list")

            running_expr = [expr for expr in expr_list.split("\n") if self.experiment_id in expr]
            print(running_expr)
            if running_expr and ("DONE" in running_expr[0]):
                stop_expr = subprocess.getoutput("nnictl stop {}".format(self.experiment_id))
                L.info(stop_expr)
                break

            elif self.experiment_id not in expr_list:
                L.info(expr_list)
                break

            else:
                if self.mfile_manage:
                    model_path = os.path.join(
                        self.experiment_path, "temp", "*_{}*".format(self.experiment_name)
                    )
                    exprs = glob.glob(model_path)
                    if len(exprs) > 3:
                        exprs.sort()
                        [shutil.rmtree(_) for _ in exprs[3:]]


    def update_tfmodeldb(self):
        """
        실험이 종료되면 모델을 DB에 저장하거나 이미 같은 이름의 모델이 존재할 시 점수를 비교하여 업데이트 합니다.
        """
        model_path = os.path.join(self.experiment_path, "temp", "*_{}*".format(self.experiment_name))
        exprs = glob.glob(model_path)
        if not exprs:
            return 0

        exprs.sort()
        exprs = exprs[0]
        metrics = os.path.basename(exprs).split("_")[:2]
        metrics = [float(metric) for metric in metrics]

        score_sql = """SELECT mae 
                       FROM atmos_model_metadata
                       WHERE model_name = '{}'
                       ORDER BY mae;""".format(self.experiment_name)
        saved_score = engine.execute(score_sql).fetchone()

        if not saved_score or (metrics[0] < saved_score[0]):
            winner_model = os.path.join(
                os.path.join(self.experiment_path, "temp", self.experiment_name)
            )
            if os.path.exists:
                shutil.rmtree(winner_model)
            os.rename(exprs, winner_model)
            
            m_buffer = zip_model(winner_model)
            encode_model = codecs.encode(pickle.dumps(m_buffer), "base64").decode()

            engine.execute(INSERT_OR_UPDATE_MODEL.format(mn=self.experiment_name, 
                                                         mf=encode_model))
            engine.execute(
                INSERT_OR_UPDATE_SCORE.format(mn = self.experiment_name,
                                              expr_id = self.experiment_id,
                                              score1 = metrics[0],
                                              score2 = metrics[1])
            )
            L.info("saved model %s %s" % (self.experiment_id, self.experiment_name))


    def modelfile_cleaner(self):
        """
        temp 폴더에 있는 모든 모델파일을 삭제합니다.
        가장 마지막에 실행하여 저장되고 남은 모델파일들을 삭제하는 용도로 사용할 수 있습니다.
        """
        model_path = os.path.join(
            self.experiment_path, "temp", "*"
        )
        exprs = glob.glob(model_path)
        [shutil.rmtree(_) for _ in exprs]
