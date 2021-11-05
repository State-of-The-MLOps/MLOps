from mlflow.entities import experiment
from prefect.run_configs.local import LocalRun
from prefect.schedules.schedules import CronSchedule
from task import etl, log_best_model, train_mlflow_ray

import prefect
from prefect import Flow, Parameter


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
            extract_query = Parameter(
                "extract_query", "SELECT * FROM insurance"
            )

            host_url = Parameter("host_url", "http://mlflow-server:5000")
            exp_name = Parameter("exp_name", "insurance")
            metric = Parameter("metric", "mae")
            model_type = Parameter("model_type", "xgboost")
            num_trials = Parameter("num_trials", 10)

            X, y = etl(extract_query)

            is_end = train_mlflow_ray(
                X, y, host_url, exp_name, metric, num_trials
            )

            if is_end:
                log_best_model(is_end, host_url, exp_name, metric, model_type)
                
        flow.run_config = LocalRun(working_dir="prefect/insurance")

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
        self.flow.schedule(CronSchedule(self._schedule))

    @property
    def flow(self):
        return self._flow

    @property
    def project_name(self):
        return self._project_name

    @property
    def flow_name(self):
        return self._flow_name
