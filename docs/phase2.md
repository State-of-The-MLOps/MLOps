# Phase2 상세

- [발표자료](https://docs.google.com/presentation/d/1TC_wMWykpN7QATgJnGuMVkwP_cm4PjEe3M3L57RuAfY/edit#slide=id.gf0d4a04c0e_2_75)

# On premise

## data & env

- [링크](https://drive.google.com/drive/folders/16BYXTck28c4Lvz8ps31atB8zfaBtHlW0?usp=sharing)에서 mnist data를 다운받아 준비합니다.
- postgres는 [phase1상태](phase1.md)와 같이 준비되어야 합니다.
- enviorment variable를 .env파일에 포함시켜 줍니다.
```plain
POSTGRES_USER=postgres
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
   * requirements.txt로 설치해본 결과 tfdv문제때문에 설치가 원활하지 않습니다.
   * 설치가 원활하지 않다면 sh파일의 명령어를 복사해서 커맨드 창에서 설치해줍니다. (ex 윈도우환경)
5. fast api server를 실행시킵니다.
   * `python main.py`
6. mlflow server를 실행시킵니다
   * `mlflow server --backend-store-uri postgresql://<user name>:<pass word>@localhost:5432/postgres --default-artifact-root <저장 경로>`
7. prefect를 실행해 줍니다.
   1. `python prefect/mnist/main.py` : mnist pipeline 추가
   2. `prefect agent local start`
   - prefect 실행시 mnist의 is_cloud 파라미터를 False로 변경해줍니다.

# k8s

## data & env

### data

- data 준비
   <details>
       <summary>google cloud storage를 쓰지 않을 경우</summary>
       
        import gdown

        google_path = 'https://drive.google.com/uc?id='
        file_id = '115LZXgZA6gPQvf5FPI1b0nsnhNz5mzH0'
        output_name = 'data_mnist_train.csv'
        gdown.download(google_path+file_id,output_name,quiet=False)
        google_path = 'https://drive.google.com/uc?id='
        file_id = '1ExfRt-4YfbP8gOAXfudlR6Lt7PbPhJzs'
        output_name = 'data_mnist_valid.csv'
        gdown.download(google_path+file_id,output_name,quiet=False)

   </details>


    <details>
    <summary>google cloud storage를 사용할 경우</summary>
    
      def insert_info():
          insert_q = """
              INSERT INTO data_info (
                  path,
                  exp_name,
                  version,
                  data_from
              ) VALUES (
                  '{}',
                  '{}',
                  {},
                  '{}'
              )
          """

          engine.execute(insert_q.format(
              'data/mnist_train.csv',
              'mnist',
              1,
              'mnist_company'
          ))
          engine.execute(insert_q.format(
              'data/mnist_valid.csv',
              'mnist',
              1,
              'mnist_company'
          ))

      insert_info()

    - google cloud storage에 choonsik-storage 이름으로 bucket생성 (다른이름일 경우 configmap.yaml 수정필요)
      - data폴더 아래에 데이터 저장 (`configmap` : CLOUD_TRAIN_MNIST: data/mnist_train.csv)
    - db에 cloud storage에 있는 data에 대한 정보 기록
    </details>

### kubernetes secret

- atmos-api-key : 온도정보를 받아오는 api key
- prefect-config : [링크](https://cloud.prefect.io/user/keys) 에서 key 발급후 ~/.prefect/config.toml 에 기록 ([참고](https://docs.prefect.io/orchestration/concepts/api_keys.html#using-api-keys))
- psql-passwd : postgresql password
- [링크](https://cloud.google.com/docs/authentication/getting-started)를 참고하여 service-account-file을 발급받고 k8s secret으로 관리

## Project

0. data&env 단계를 수행합니다.
1. `cd k8s && kubectl apply -k kustomization.yaml`

참고: [frontend](https://github.com/ehddnr301/mnist_test_FE)
# Review
