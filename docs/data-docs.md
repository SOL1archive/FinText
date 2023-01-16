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

# Dataset


# Dataloader