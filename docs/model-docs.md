# FinTextModel

## Model Diagram
![](./NLP%20Architecture.png)

## Description
모델의 생성자(`__init__` 함수) 내 `Neural Network` 각주 밑 부분을 수정하여 하이퍼 파라미터를 수정할 수 있음. 

### Neural Network Architecture
> Note. 아래 실제 변수들은 클래스 객체들의 변수들로 호출할 때 `self` 키워드가 붙지만 모두 공통되므로 문서상에서는 생략함.

- `article_cnn` & `community_cnn`: `article_cnn`은 기사 데이터의 임베딩 행렬에 대한 CNN, `community_cnn`은 커뮤니티 데이터의 임베딩 행렬에 대한 CNN임. 둘 다 임의로 한개의 층으로 구성했지만 필요에 따라 여러 층으로 구성할 수 있음. 현실적으로 여러 층으로 구성해야 할 가능성이 높음. 기본적으로는 `BatchNorm`, `Residual Connection`이 적용되어 있지 않음. 여러 학습 현황에 따라 면밀하게 적용하면 좋음.
- `total_ffn`: 각 입력에 따라 처리한 후 최종 처리를 하는 신경망. 임의로 입력 피처 크기를 정했지만 이전 신경망들의 출력 벡터의 크기에 따라 **하이퍼파라미터를 조정할 필요가 있음.** 기본적으로는 Normalization, Regularization 기법이 적용되어있지 않기 때문에 학습에 따라 `Dropout`, `BatchNorm` 등을 적용하면 좋음.
- `softmax`: 최종적으로 결정을 출력하는 계층. 출력 벡터의 크기는 **정답 벡터의 크기와 동일**하게 설정하면 됨. 기본적으로는 4로 설정.

## Method
- `embed_text(text_lt, tokenizer, model) -> torch.tensor`: \
    텍스트들의 리스트, 토크나이저, 모델을 입력받아 임베딩 후 모든 행렬들을 쌓아 행렬(2차원 텐서)로 반환함.
- `dim_fix(self, tensor, row_len) -> torch.tensor`: \
    행렬의 행의 길이가 `row_len`과 다를 때 행의 길이를 `row_len`에 맞춤. 행의 길이가 `row_len`보다 적을 경우 0 padding을 수행하고, 많을 경우 차원 축소 기법을 사용함. `config`를 통해 차원 축소 기법을 지정할 수 있음. 지원하는 차원 축소 기법으로는 `SVD`, `PCA`, `NMF`가 있음.

## 생성자 파라미터
- `config`: `dict` \
    하이퍼파라미터들의 딕셔너리. 주피터 환경에서 쉬운 테스트를 위함. \
    필수 원소)
    - `article_row_len` & `community_row_len` : 고정된 행의 길이 \
        통계적으로 크기를 살펴본 후 **적절하게 조정될 필요가 있음.**
    - `decomposition_method`: 행의 길이가 설정된 길이보다 더 길 경우 적용하는 차원 축소 기법의 종류. `SVD`, `PCA`, `NMF`가 있음. 자연어 데이터 분포에 따라 적절하게 조정될 필요가 있음.
