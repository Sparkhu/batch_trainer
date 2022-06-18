# batch trainer

해당 프로젝트는 일정 기간 통합된 시계열 데이터를 배치학습하여 배치 모델을 생성합니다.




## 1. 주요 기능
1. 시계열 데이터를 Sliding Window로 구성하여 LSTM 모델을 스파크 클러스터에서 배치 학습합니다.
2. **HDFS**에 /year='YYYY'/month='mm'/day='dd' 형식 아래 파티션된 데이터를 사용합니다.
3. 파라미터로 입력된 [DB, Table, FeatureName] 정보에 따라 해당 Feature를 학습합니다.
4. 학습된 모델은 **HDFS** `/modelDir/offline/{DB}_{Table}_{FeatureName}.pt 형식으로 저장됩니다.

## 2. 시작하기 전 준비사항
이 프로젝트는 Python 3.7 버전 및 Hadoop과 Yarn Cluster 위에서 동작하는 Spark Cluster모드로 진행하였습니다.<br><br>

또한 LSTM모델을 Spark환경에서 수행하기 위해 BigDL 프로젝트를 적용하였기 때문에 conda 가상환경 구성이 필수적입니다. [다음을 참조하세요](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/hadoop.html).<br><br>

따라서 다음과 같은 의존성을 지닙니다.
- conda
- Python == 3.7 //3.8 이상 버전에서 LSTM모델 버그가 존재합니다.
- Hadoop >= 3.3.1
- Spark >= 3.0.3
- Python package bigdl-orca, torch==1.7.1, torchvision==0.8.2, six, cloudpickle, jep==3.9.0


## 3. 시작하기
### 3.1 프로젝트 클론
다음과 같이 프로젝트를 받아주세요.
``` sh
$ mkdir {YOUR_DESIRED_PATH} && cd {YOUR_DESIRED_PATH}
$ git clone 
$ cd batch_trainer
```
### 3.2 가상환경 구성
Spark cluster 구성에 Python environment를 각 Worker에 전달하기 위해서 **conda environment를 Pacakge**해줘야 빠른 속도로 Cluster 구성을 진행할 수 있습니다.
``` sh
$ conda create -n {YOUR_VENV_NAME} python=3.7 # Python must be 3.7
$ conda activate {YOUR_VENV_NAME}
$ pip install bigdl-orca torch==1.7.1 torchvision==0.8.2 six cloudpickle jep==3.9.0
$ conda pack -o environment.tar.gz # Recommended
```
### 3.3 배치 학습
``` sh
$ spark=submit-with-bigdl \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory {ADJUST_FOR_YOU} \
    --driver-memory {ADJUST_FOR_YOU} \
    --executor-cores {ADJUST_FOR_YOU} \
    --num-executors {ADJUST_FOR_YOU} \
    --archives environment.tar.gz#environment \
    batch_trainer.py
```

## 4. 라이센스
This project is licensed under the terms of the [**APACHE LICENSE, VERSION 2.0**](https://www.apache.org/licenses/LICENSE-2.0.txt).
