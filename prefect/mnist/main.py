from Pipeline import Pipeline
from prefect.schedules.clocks import CronClock

from prefect import Client

if __name__ == "__main__":
    Client().create_project(project_name="mnist")
    pipeline = Pipeline("mnist", "mnist_flow")
    pipeline.create_flow()
