import prefect
import pandas as pd
from datetime import timedelta
from prefect import Flow
from task import *

class atmos_ETL:
    _project_name = None
    _flow_name = None
    _logger = None
    _flow = None

    def __init__(self, project_name, flow_name, schedule=None):
        self._conn = connect('postgres')
        self._logger = prefect.context.get("logger")
        self._project_name = project_name
        self._flow_name = flow_name
        self._schedule = schedule

        self.start_date = self._conn.execute(
            "SELECT time FROM atmos_stn108 ORDER BY time DESC;"
            ).fetchone()[0]
        self.start_date = pd.to_datetime(self.start_date) + timedelta(hours=1)
        self.end_date = pd.Timestamp.utcnow() + timedelta(hours=9)


    def create_flow(self):
        self._logger.info(f"Create {self._flow_name} flow")
    
        with Flow(self._flow_name) as flow:
            self._logger.info('start data extract')
            extr_result = data_extract(self._conn,
                                       self.start_date,
                                       self.end_date)
            
            if extr_result[0]:
                self._logger.info('success data extract')
                valid_result = data_validation(self._conn,
                                               self.start_date,
                                               extr_result[0])
            else:
                self._logger.info(extr_result[1])
                return False

            if valid_result[0]:
                self._logger.info(f"statistics are the same as origin data")
                data_load_to_db(self._conn, valid_result[1])
            else:
                self._logger.info(f"There is a data drift")
                self._logger.info(self._conn,
                            valid_result[1])                
                return False
        
        self._flow = flow
        self._register()
        

    def _register(self):
        self._logger.info(
            f"Regist {self._flow_name} flow to {self._project_name} project"
        )
        self._logger.info(f"Set Cron {self._schedule}")

        self._flow.register(
            project_name=self._project_name, 
            idempotency_key=self.flow.serialized_hash()
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
    