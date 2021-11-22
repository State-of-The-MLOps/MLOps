# Phase2 local에서 시작하기

## data & env

- [링크](https://drive.google.com/drive/folders/16BYXTck28c4Lvz8ps31atB8zfaBtHlW0?usp=sharing)에서 mnist data를 다운받아 준비합니다.
- postgres는 [phase1상태](phase1.md)와 같이 준비되어야 합니다.
- enviorment variable를 .env파일에 포함시켜 줍니다.
```plain
POSTGRES_PASSWORD=0000
POSTGRES_SERVER=localhost
POSTGRES_PORT=5431
POSTGRES_DB=postgres
MLFLOW_HOST=http://localhost:5000
TRAIN_MNIST=/절대/경로/mnist_train.csv
VALID_MNIST=/절대/경로/mnist_valid.csv
```

## project

0. data&env 단계를 수행합니다.
1. 소스코드를 다운받습니다.
2. `conda create --name mlops-phase1 python=3.8`
3. `conda activate mlops-phase1`
4. `bash requirements.sh`로 필요한 라이브러리를 설치합니다.
   * requirements.txt로 설치해본 결과 tfdv문제때문에 설치가 원할하지 않습니다.
5. fast api server를 실행시킵니다.
   * `python main.py`
6. mlflow server를 실행시킵니다
   * `mlflow server --backend-store-uri postgresql://postgres:0000@localhost:5432/postgres --default-artifact-root <저장 경로>`
7. prefect를 실행해 줍니다.
   1. `python prefect/mnist/main.py` : mnist pipeline 추가
   2. `prefect agent local start`
   - prefect 실행시 mnist의 is_cloud 파라미터를 False로 변경해줍니다.
