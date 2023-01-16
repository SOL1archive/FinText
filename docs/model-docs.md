# FinTextModel
- hyperparameters: dict \
    하이퍼파라미터들의 딕셔너리. 주피터 환경에서 쉬운 테스트를 위함. \
    필수 원소)
    - `article_row_len` & `community_row_len` : 고정된 행의 길이 \
        통계적으로 크기를 살펴본 후 **적절하게 조정될 필요가 있음.**
    - `decomposition_method`: 행의 길이가 설정된 길이보다 더 길 경우 적용하는 차원 축소 기법의 종류. `SVD`, `PCA`, `NMF`가 있음. 자연어 데이터 분포에 따라 적절하게 조정될 필요가 있음.

# Model Diagram
![](./NLP%20Architecture.png)