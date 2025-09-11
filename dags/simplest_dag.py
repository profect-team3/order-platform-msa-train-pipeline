import pendulum
from airflow.sdk import dag, task


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def example_simplest_dag():
    @task
    def my_task():
        print("Hello from simplest DAG")

    my_task()


# Invoke the DAG
example_simplest_dag()
