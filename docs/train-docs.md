# TrainTestApp
## Description
학습 및 테스트에 대한 다양한 기능을 제공하는 클래스. 자체 로그 기록 기능, `tensorboard` 표현 기능, 데이터셋, 데이터로더 생성, 모델 생성 및 관리, 학습시 사용 모델, 최적화 알고리즘 관리 등 다양한 부분을 조정할 수 있음. Jupyter Notebook을 사용할 경우에도 쉽게 코드를 재사용할 수 있음.

로그 기록은 기본적으로 "./log/train-test.log" 에 저장됨.

## Method
- `__init__(**config) -> None`: \
    생성자. argument 키워드들을 대입하여 하이퍼 파라미터를 조정할 수 있음. \
    하이버 파라미터 목록)
    - `num_epoch: int` \
        기본값 30. 에포크 횟수 지정
    - `lr: float` \
        학습률. 학습률 감쇠는 아래 `lr_scheduler`를 통해 설정할 수 있음.
    - `lr_scheduler: callable(epoch: int)`:
        학습률 감쇠 함수를 설정할 수 있음. `lambda` 식과 일반 함수 모두를 지원함. 함수의 입력 파라미터는 `epoch`임.
    - `df_list: pd.DataFrame` \
        데이터프레임의 리스트. 데이터프레임들의 리스트를 입력받아 데이터셋과 데이터로더를 생성하는 데 사용함.
- `prepare_dataset`: \
    입력받은 `df_list`들과 `train_size`를 바탕으로 훈련, 테스트용 데이터셋을 생성함. 만약 전체 데이터셋을 GPU에 올릴 수 있는 경우 주석을 해제하면 더 빠른 학습이 가능.
- `prepare_dataloader`: \
    생성한 훈련, 테스트 데이터셋으로 훈련, 테스트 데이터로더를 생성함.
- `prepare_model`: \
    모델을 생성함. 여러 GPU를 병렬로 사용하는 경우, 병렬 처리가 가능하도록 모델을 생성함.
- `prepare_tensorboard_writer`: \
    `tensorboard`에 대한 훈련, 테스트용 `SummaryWriter`를 생성함. 훈련용 `SummaryWriter`인 `train_writer`와 테스트용 `SummaryWriter`인 `test_writer`를 생성함.
- `prepare_optimizer`: \
    최적화 알고리즘 객체 `optimizer: optim.Adam`과 비용함수에 따른 학습률 스케줄러인 `scheduler:  optim.lr_scheduler`를 생성함. 최적화 알고리즘의 경우 일반적으로 `Adam`이 가장 좋은 성능을 낼 것으로 기대할 수 있으므로, 기본 설정함. 만약 바꾸고 싶다면 해당 코드를 수정하면 됨.
- `train`: \
    학습을 수행하는 함수. `num_epoch`에 따라 해당 횟수 많큼 Epoch를 수행함. 매 Epoch 마다 Validation Test를 수행함. 모든 결과는 로그로 기록되고 `tensorboard`에서도 학습 상황을 확인할 수 있음.
- `save`: \
    학습 후 모델을 저장함. 저장 이름은 저장 시간임. 기본적으로 `model-dir` 폴더에 모델을 저장함.
- `main`: \
    이 모든 함수를 한번에 수행하고 마지막에 모델을 저장할지 여부를 묻는 함수. 학습이 끝나면 `Ctrl + C`를 통해 종료할 수 있음.
