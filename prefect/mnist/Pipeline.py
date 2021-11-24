from prefect.run_configs import LocalRun
from prefect.schedules.schedules import CronSchedule
from task import (
    case2,
    log_experiment,
    make_feature_weight,
    train_knn,
    tune_cnn,
)

import prefect
from prefect import Flow, Parameter, case


class Pipeline:
    _project_name = None
    _flow_name = None
    _logger = None
    _flow = None

    """
        _param1 = Parameter("data_path", default="default_path")
        _param2 = Parameter("model_name", default="GPN")
    """

    def __init__(self, project_name, flow_name, schedule=None):
        self._logger = prefect.context.get("logger")
        self._logger.info("Create Pipeline")

        self._project_name = project_name
        self._flow_name = flow_name
        self._schedule = schedule

    def create_flow(self):
        self._logger.info(f"Create {self._flow_name} flow")
        with Flow(self._flow_name) as flow:
            """

            data = load_data(self._param1)
            prep_data = preprocess(data)
            model = train(self._param2, prep_data)
            save_model(model)

            """

            host_url = Parameter("host_url", "http://mlflow-service:5000")
            exp_name = Parameter("exp_name", "mnist")
            metric = Parameter("metric", "loss")
            num_samples = Parameter("num_samples", 1)
            max_num_epochs = Parameter("max_num_epochs", 1)
            is_cloud = Parameter("is_cloud", True)
            data_version = Parameter("data_version", 3)

            results = tune_cnn(
                num_samples, max_num_epochs, is_cloud, data_version, exp_name
            )
            is_end = log_experiment(
                results, host_url, exp_name, metric, data_version, is_cloud
            )

            with case(is_end, True):
                feature_weight_df = make_feature_weight(
                    results, "cpu", is_cloud, data_version, exp_name
                )
                train_knn(feature_weight_df, metric, exp_name)

            with case(is_end, False):
                case2()
        flow.run_config = LocalRun(working_dir="prefect/mnist")
        self._flow = flow
        self._register()

    def _register(self):
        self._logger.info(
            f"Regist {self._flow_name} flow to {self._project_name} project"
        )
        self._logger.info(f"Set Cron {self._schedule}")

        self._flow.register(
            project_name=self._project_name,
            idempotency_key=self.flow.serialized_hash(),
        )

        if self._schedule:
            self._set_cron()

    def _set_cron(self):
        self.flow.schedule((self._schedule))

    @property
    def flow(self):
        return self._flow

    @property
    def project_name(self):
        return self._project_name

    @property
    def flow_name(self):
        return self._flow_name
