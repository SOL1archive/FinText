# Data Columns
| Name            | Type        | Description                    |
| --------------- | ----------- | ------------------------------ |
| `StartDateTime` | `datetime`  | 날짜 구간 시작 시각             |
| `EndDateTime`   | `datetime`  | 날짜 구간 끝 시각               |
| `ArticleText`    | `list(str)` | 구간 내 기사들의 리스트         |
| `CommunityText` | `list(str)` | 구간 내 커뮤니티 글들의 리스트   |
| `MetricIndex`   | `list(str)` | 커뮤티니 글들에 대한 평가 지표   |
| `Open`          | `float32`   | 시가                           |
| `High`          | `float32`   | 고가                           |
| `Low`           | `float32`   | 저가                           |
| `Close`         | `float32`   | 종가                           |

> Note. `CommunityText`와 그에 대응되는 `MetricIndex`의 원소의 순서는 동일해야 한다.

# FinTextDataset
> Note. 초기 코드에서는 아래 내용은 모델의 일부였지만 데이터를 준비하는 코드에 가까우므로 특성상 `Dataset`에 역할을 부담하는 것이 적절하다고 판단하여 `Dataset`으로 옮김.

## 생성자 파라미터
- `config`: `dict` \
    하이퍼파라미터들을 인수로 대입가능. 다음 변수들을 대입함으로써 세부 설정이 가능함. 주피터 환경에서 쉬운 테스트를 위함. \
    - `article_row_len` & `community_row_len` : 고정된 행의 길이 \
        통계적으로 크기를 살펴본 후 **적절하게 조정될 필요가 있음.**
    - `decomposition_method`: 행의 길이가 설정된 길이보다 더 길 경우 적용하는 차원 축소 기법의 종류. `SVD`, `PCA`, `NMF`가 있음. 자연어 데이터 분포에 따라 적절하게 조정될 필요가 있음.
    - `bundle_size`: 한 묶음의 크기

## Method
- 일반 함수
    - `to(self, device) -> None`: \
        모든 데이터를 `device`로 옮김.
    - `train_test_split(train_size=0.80) -> (train: FinTextDataset, test: FinTextDataset)`: \
        `train_size` 비율 대로 학습 데이터와 테스트 데이터를 분리함.
    - `__len__() -> int`: \
        전체 데이터 개수를 반환함.
    - `__getitem__(index) -> (feature_df: pd.DataFrame, target_tensor: torch.Tensor)`: \
        데이터 하나를 반환함.
- 내부함수 (외부에서 사용불가)
    - `embed_text(text_lt, tokenizer, model) -> torch.tensor`: \
        텍스트들의 리스트, 토크나이저, 모델을 입력받아 임베딩 후 모든 행렬들을 쌓아 행렬(2차원 텐서)로 반환함.
    - `dim_fix(self, tensor, row_len) -> torch.tensor`: \
        행렬의 행의 길이가 `row_len`과 다를 때 행의 길이를 `row_len`에 맞춤. 행의 길이가 `row_len`보다 적을 경우 0 padding을 수행하고, 많을 경우 차원 축소 기법을 사용함. `config`를 통해 차원 축소 기법을 지정할 수 있음. 지원하는 차원 축소 기법으로는 `SVD`, `PCA`, `NMF`가 있음.

# FinTextDataLoader
초기 변수 입력이 불편할 경우를 위해 만들어놓음. 따라서 데이터로더를 생성시 번거롭게 모든 변수를 다 입력할 필요가 없음.
