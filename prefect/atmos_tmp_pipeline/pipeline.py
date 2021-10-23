import prefect
import pandas as pd
from datetime import timedelta
from prefect import Flow
from task import *
from prefect.schedules import Schedule
from prefect.schedules.clocks import CronClock


class atmos_ETL:
    _project_name = None
    _flow_name = None
    _logger = None
    _flow = None
    

    def __init__(self, project_name, flow_name, schedule=None):
        self._logger = prefect.context.get("logger")
        self._project_name = project_name
        self._flow_name = flow_name
        self._schedule = schedule

    def create_flow(self):
        self._logger.info(f"Create {self._flow_name} flow")

        with Flow(self._flow_name) as flow:
            self._logger.info('start data extract')
            
            host_url = os.getenv('MLFLOW_HOST')
            exp_name = "atmos_tmp"
            metric = "mae"
            model_type = "tensorflow"
            num_trials = 10

            extr_result = data_extract(os.getenv('ATMOS_API_KEY'))
            valid_result = data_validation(extr_result[1])
            load_data = data_load_to_db(valid_result[1],
                            os.getenv("POSTGRES_USER"),
                            os.getenv("POSTGRES_SERVER"),
                            os.getenv("POSTGRES_PASSWORD"))
            is_end = train_mlflow_ray(load_data,
                                      host_url,                                      
                                      exp_name,
                                      metric,
                                      num_trials)
            log_best_model(is_end, host_url, exp_name, metric, model_type)

        self._flow = flow
        self._register()
        

    def _register(self):
        self._logger.info(
            f"Regist {self._flow_name} flow to {self._project_name} project"
        )
        self._logger.info(f"Set Cron {self._schedule}")

        if self._schedule:
            self._set_cron()

        self._flow.register(
            project_name=self._project_name, 
            idempotency_key=self.flow.serialized_hash()
        )

        
    def _set_cron(self):
        schedule = Schedule(clocks=[CronClock(self._schedule)])
        self._flow.schedule = schedule
        

    @property
    def flow(self):
        return self._flow

    @property
    def project_name(self):
        return self._project_name

    @property
    def flow_name(self):
        return self._flow_name
    