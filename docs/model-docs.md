# FinTextModel

## Model Diagram
![](./NLP%20Architecture.png)

## Description
- 모델의 생성자(`__init__` 함수) 내 `Neural Network` 각주 밑 부분을 수정하여 하이퍼 파라미터를 수정할 수 있음. 
- 입력은 미니배치를 기준으로 함.

## Neural Network Architecture
> Note. 아래 실제 변수들은 클래스 객체들의 변수들로 호출할 때 `self` 키워드가 붙지만 모두 공통되므로 문서상에서는 생략함.

- `article_cnn` & `community_cnn`: `article_cnn`은 기사 데이터의 임베딩 행렬에 대한 CNN, `community_cnn`은 커뮤니티 데이터의 임베딩 행렬에 대한 CNN임. 둘 다 임의로 한개의 층으로 구성했지만 필요에 따라 여러 층으로 구성할 수 있음. 현실적으로 여러 층으로 구성해야 할 가능성이 높음. 기본적으로 `BatchNorm`은 적용되어 있고, `Residual Connection`은 적용되어 있지 않음. 여러 학습 현황에 따라 면밀하게 적용하면 좋음.
- `gru`: `nn.GRU`계층과 `nn.Linear` 서브 계층으로 구분되어 있음.
- `total_ffn`: 각 입력에 따라 처리한 후 `nn.Flatten`으로 쌓아 최종 처리를 하는 신경망. 임의로 입력 피처 크기를 정했지만 이전 신경망들의 출력 벡터의 크기에 따라 **입력층 개수를 조정할 필요가 있음.** 기본적으로는 Normalization, Regularization 기법이 적용되어있지 않기 때문에 학습에 따라 `Dropout`, `BatchNorm` 등을 적용하면 좋음.

        텐서의 출력 크기을 구하는 방법) \
        텐서의 각 출력 차원에 대해,
        
        $$
        \frac{\text{input size} - \text{kernel size} + 2\ \text{padding}}{\text{stride}} + 1
        $$

        python 인터프리터에 대입할 때 다음 식을 사용하면 좋음
        ```python
        int((input_size - kernel_size + 2 * padding) / stride) + 1
        ```
해당 식을 사용하여 텐서의 각 차원에 대한 출력 크기를 구하고 모두를 곱해서 Flatten 후의 최종적인 CNN층의 출력 크기를 쉽게 구할 수 있음.

- `softmax`: 최종적으로 결정을 출력하는 계층. 출력 벡터의 크기는 **정답 벡터의 크기와 동일**하게 설정하면 됨. 기본적으로는 4로 설정.
