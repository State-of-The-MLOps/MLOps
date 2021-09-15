# API List

API_URL : 127.0.0.1

- [API List](#api-list)
    - [Train](#train)
        - [Insurance](#insurance훈련)
    - [Prediction](#predict)
        - [Insurance](#insurance예측)
        - [Temperature](#Temperature예측)


## Train

### Insurance훈련

#### 요청

```
PUT {{API_URL}}/train/insurance
```

| 파라미터        | 파라미터 유형 | 데이터 타입 | 필수 여부 | 설명                    |
| --------------- | ------------- | ----------- | --------- | ----------------------- |
| `PORT`      | `body`        | `int`    | `(default) 8080`        | NNi 포트번호      |
| `experiment_sec`      | `body`        | `int`   | `(default) 20`        | 학습시간(초)            |
| `experiment_name`      | `body`        | `(default) exp1`    | ✅        | 학습이름    |
| `experimenter` | `body`        | `str`    | `(default) DongUk`        | 연구자 이름 |
| `model_name`      | `body`        | `str`    | `(default) insurance_fee_model`        | 학습 모델 이름                    |
| `version`          | `body`        | `float`    | `(default) 0.1`        | 모델 버전        |


<br/>

#### 응답

| 키             | 데이터 타입 | 설명          |
| -------------- | ----------- | ------------- |
| `msg`  | `string`    | NNi실험을 확인할 수 있는 링크주소    |
| `error`  | `string`    | 에러내용    |


```jsonc
{
  "result": 'Check out {{API_URL}}:{PORT}',
}
{
  "error": "Error info"
}
```


### Temperature훈련

#### 요청

| 파라미터        | 파라미터 유형 | 데이터 타입 | 필수 여부 | 설명                    |
| --------------- | ------------- | ----------- | --------- | ----------------------- |
| `expr_name`      | `body`        | `string`    | ✅        | 학습이름      |


<br/>

#### 응답

| 키             | 데이터 타입 | 설명          |
| -------------- | ----------- | ------------- |
| `error`  | `string`    | 에러내용    |

```jsonc
{
  "error": "Error info"
}
```

## Predict

### Insurance예측

#### 요청

```
PUT {{API_URL}}/predict/insurance
```

| 파라미터        | 파라미터 유형 | 데이터 타입 | 필수 여부 | 설명                    |
| --------------- | ------------- | ----------- | --------- | ----------------------- |
| `age`      | `body`        | `int`    | ✅        | 나이      |
| `sex`      | `body`        | `int`   | ✅        | 성별            |
| `bmi`      | `body`        | `float`    | ✅        | bmi수치    |
| `children` | `body`        | `int`    | ✅        | 자녀 수 |
| `smoker`      | `body`        | `int`    | ✅        | 흡연여부                    |
| `region`          | `body`        | `int`    | ✅        | 거주지역        |

<br/>

#### 응답

| 키             | 데이터 타입 | 설명          |
| -------------- | ----------- | ------------- |
| `result`  | `float`    | 예측된 보험료 값    |
| `error`  | `string`    | 에러내용    |


```jsonc
{
  "result": 3213.123,
}
{
  "error": "Error info"
}
```


### Temperature예측

#### 요청

```
PUT {{API_URL}}/predict/atmos
```

| 파라미터        | 파라미터 유형 | 데이터 타입 | 필수 여부 | 설명                    |
| --------------- | ------------- | ----------- | --------- | ----------------------- |
| `time_series`      | `body`        | `List[float]`    | ✅        | 72일간의 온도데이터      |


<br/>

#### 응답

| 키             | 데이터 타입 | 설명          |
| -------------- | ----------- | ------------- |
| x  | `List[float]`    | 예측된 향후 24일간 온도값    |
| `error`  | `string`    | 에러내용    |


```jsonc
{
  [32.32, 33.32, 34.11...]
}
{
  "error": "Error info"
}
```
