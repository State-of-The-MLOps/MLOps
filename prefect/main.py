from pipeline import atmos_ETL
from prefect import Client


if __name__ == '__main__':
    Client().create_project(project_name='atmos_test')
    Pipeline = atmos_ETL("atmos_test", "test_flow", "0 5 * * *")
    Pipeline.create_flow()