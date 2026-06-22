# MDP 모델링과 DQN을 활용한 대화형 상품 추천 시스템

본 프로젝트는 2026년 1학기 **순차적 의사결정** 강의의 term project로 진행한 실험입니다.  
대화형 추천 시스템을 finite-horizon Markov Decision Process(MDP)로 모델링하고, Deep Q-Network(DQN)을 활용하여 사용자 선호를 순차적으로 파악하며 상품을 추천하는 정책을 학습합니다.

## 프로젝트 구조
```
DQN-CRS/
├── data/                         # 원본 및 전처리 데이터 저장 폴더
│   ├── items.csv                  # 무신사 상품 데이터
│   ├── reviews.csv                # 무신사 리뷰 데이터
│   ├── processed_items.csv        # 전처리된 상품 데이터
│   └── processed_interactions.csv # 전처리된 user-item interaction 데이터
│
├── results/                       # 학습 및 평가 결과 저장 폴더
│   ├── checkpoints/               # 학습된 DQN 모델 checkpoint
│   │   └── dqn_crs.pt
│   ├── metrics/                   # 학습 및 평가 지표
│   │   ├── train_metrics.csv
│   │   ├── evaluation_raw.csv
│   │   ├── evaluation_summary.csv
│   │   └── evaluation_failure.csv
│   └── plots/                     # 학습 곡선 시각화 결과
│       ├── train_reward.png
│       └── train_success.png
│
├── src/                           # 실험 코드
│   ├── config.py                  # 데이터 경로, MDP/RL 설정값, reward 설정
│   ├── preprocess.py              # 상품 및 리뷰 데이터 전처리
│   ├── env.py                     # CRS-MDP 환경 구현
│   ├── dqn_model.py               # DQN 네트워크 정의
│   ├── replay_buffer.py           # Experience replay buffer 구현
│   ├── train.py                   # DQN 학습 코드
│   └── evaluate.py                # Random, Heuristic, DQN 정책 평가 코드
│
├── .gitignore                     # data/, __pycache__ 등 제외
└── README.md
```

## 실행 방법

### 1. 프로젝트 폴더로 이동
```bash
cd DQN-CRS
```

---

### 2. 데이터 준비
`data/` 폴더는 `.gitignore`에 포함되어 있으므로 GitHub에는 업로드하지 않는다.  
코드를 실행하려면 로컬 환경에서 다음 파일을 직접 준비해야 한다.

```text
DQN-CRS/
└── data/
    ├── items.csv
    └── reviews.csv
```

---

### 3. 데이터 전처리
처음 실행하거나, 원본 데이터 또는 사용할 속성 구성이 바뀐 경우 전처리를 수행한다.

```bash
python src/preprocess.py
```

전처리 결과로 다음 파일이 생성된다.

```text
data/processed_items.csv
data/processed_interactions.csv
```

---

### 4. DQN 학습
```bash
python src/train.py
```

학습 결과는 다음 위치에 저장된다.

```text
results/checkpoints/dqn_crs.pt
results/metrics/train_metrics.csv
results/plots/train_reward.png
results/plots/train_success.png
```

---

### 5. 정책 평가

```bash
python src/evaluate.py
```

평가에서는 다음 세 가지 정책을 비교한다.

| Policy | 설명 |
|---|---|
| Random | 가능한 action 중 무작위 선택 |
| Heuristic | 정해진 순서대로 질문하고 후보가 줄어들면 추천 |
| DQN | 학습된 DQN 모델 기반 정책 |

평가 결과는 다음 위치에 저장된다.

```text
results/metrics/evaluation_raw.csv
results/metrics/evaluation_summary.csv
results/metrics/evaluation_failure.csv
```