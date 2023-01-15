# FinTextModel
- hyperparameters: dict \
    하이퍼파라미터들의 딕셔너리. 주피터 환경에서 쉬운 테스트를 위함. \
    필수 원소)
    - `community` & `article`
        - `cnn_out_channels`: CNN의 커널 개수
        - `kernel_size`: 커널의 크기. **대칭형 커널의 경우에도 2-튜플 사용**. 홀수를 사용할 것을 권장함.
        - `stride`: stride. 일반적으로 1로 고정
        - `same`: `True`일 경우 동일 합성곱(Same Convolution)을 진행.

# Model Diagram
![](./NLP%20Architecture.png)