<h1 align='center'>
MLOps
</h1>
<h2 align='center'>
👊 Build MLOps system step by step 👊
</h2>
<div align='center'>
<img src="https://img.shields.io/badge/python3.8-007396?style=for-the-badge&logo=python&logoColor=white">
</div>

<div align='center'>
<h3>
🚀 프로젝트 목적 🚀
</h3>
<h3>
지속가능한 AI 서비스를 위한 MLOps 구성
</h3>
</div>

## 패치노트

### Phase2

```

├── README.md
├── app
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── data_class.py
│   │   └── router
│   │       ├── __init__.py
│   │       ├── predict.py
│   │       └── train.py [deprecated]
│   ├── database.py
│   ├── query.py
│   ├── schema.py
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── experiments [deprecated]
├── logger.py
├── main.py
├── prefect
│   ├── atmos_tmp_pipeline
│   │   ├── main.py
│   │   ├── pipeline.py
│   │   ├── query.py
│   │   ├── task.py
│   │   └── utils.py
│   ├── insurance
│   │   └── ...
│   └── mnist
│       └── ...
├── pyproject.toml
└── requirements.txt
```

- [Phase2 상세내용](./docs/phase2.md)
- Phase2 단계에서는 MLOps1단계와 자동화 배포 단계를 다루고있습니다.
- train은 더이상 api요청을 통해 이루어지지 않고 prefect를 통해 관리됩니다.
  - atmos_tmp 데이터의 경우 매일 데이터 크롤링하기 위해 scheduling된 작업을 수행합니다.
  - prefect cloud를 통해 수동으로 실험을 실행시켜 더나은 모델을 위한 training을 수행할 수 있습니다.
  - Phase1 단계에서 사용한 NNi의 역할을 Ray tune을 통해 진행합니다. [변경이유](./docs/shoveling_note.md)
  - 모델을 더이상 query문을 작성해서 조회 및 업데이트되지 않고 mlflow를 통하여 관리됩니다.
- predict는 기존에 phase1 단계에서 모델을 미리로드, 매번 로드 하는 방식을 개선하였습니다.
  - 모델은 prediction 요청이 들어왔을 때만 로드되며 로드된 모델은 정해진 시간만큼 캐싱되어 사용됩니다.
  - 이를 위해 redis를 고려하였으나 최종적으로는 사용하지 않습니다. [변경이유](./docs/shoveling_note.md)

### Phase1

```
├── app
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── router
│   │   │   ├── __init__.py
│   │   │   ├── predict.py
│   │   │   └── train.py
│   │   └── schemas.py
│   ├── database.py
│   ├── models.py
│   ├── query.py
│   └── utils.py
├── docs
│   └── api-list.md
├── experiments
│   ├── atmos_tmp_01
│   │   ├── config.yml
│   │   ├── preprocessing.py
│   │   ├── search_space.json
│   │   └── train.py
│   ├── expr_db.py
│   └── insurance
│       ├── config.yml
│       ├── query.py
│       ├── search_space.json
│       └── trial.py
├── logger.py
├── main.py
├── pyproject.toml
└── requirements.txt
```

- [Phase1 상세내용](./docs/phase1.md)
- Phase1에서는 MLOps 0단계 구성을 위해 노력했습니다.
- train은 experiments 폴더에 구성되어 있습니다.
  - 본래 0단계에서는 수동 ML형태를 취합니다.
  - 본 프로젝트에서는 어느정도의 자동화된 모습을 구현하기 위해 train을 api형태로 요청할 수 있게 구성하였습니다.
  - train 요청에따라 subprocess로 NNi를 이용한 hyper parameter tuning을 진행합니다.
  - 각 실험결과 best모델을 현재 저장된 모델 성능과 비교하여 db에 직렬화시켜 저장합니다.
- predict는 `api-router-predict.py` 에 구성되어 있습니다.
  - prediction요청에 따라 결과를 반환합니다.
  - temp예측의 경우 서버 시작시 모델을 로드하여 모델을 매번 읽어오지 않도록 합니다.
  - insurance예측의 경우 db에서 매번 모델을 읽어와 예측을 진행합니다.
- logger
  - 요청, 실험진행 등을 log로 남깁니다.
