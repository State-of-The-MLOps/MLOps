# Phase1 상세

- [발표자료](https://docs.google.com/presentation/d/16cQSK4t3O86uMFg6iEr02MrtNUQf95RTcTleAAR44fY)

## How to enter this project

### data & env

- postgres db 를 준비합니다.
  - `docker run -d --name postgre -p 5432:5432 -e POSTGRES_PASSWORD=<postgres password> postgres:13.4` 
- [Data](https://drive.google.com/file/d/1YPOPA1jnXFyJvl6ikThejvVnxOJl9ya5/view?usp=sharing)를 다운로드 받아 postgresql db에 넣어줍니다.
  - `docker cp </폴더/경로/postgres_20211026.sql> <container_name>:/postgres_20211026.sql`
  - `docker exec -it <containerID> bash`
  - `psql postgres < /postgres_20211026.sql`
    - 만약 컨테이너에서 role "root" does not exist 에러가 난다면 `su -l postgres` 로 유저를 변경한 후에 작업해 주세요
- enviornment variable 를 .env파일에 포함시켜 줍니다.
  ```plain
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=0000
    POSTGRES_SERVER=localhost
    POSTGRES_PORT=5432
    POSTGRES_DB=postgres
  ```

### Project

0. data&env 단계를 수행합니다.
1. [Phase1](https://github.com/State-of-The-MLOps/MLOps/releases/tag/v1.0.0) 에서 Source코드를 다운받습니다.
2. `conda create --name mlops-phase1 python=3.8`
3. `conda activate mlops-phase1`
4. `pip install -r requirements.txt` 로 필요한 라이브러리를 설치합니다.
5. `python main.py` 로 서버를 실행시킵니다.
6. http://localhost:8000/docs 에서 fastapi swagger를 통해 api를 테스트합니다.

## Review