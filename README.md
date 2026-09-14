# Customer Churn — Ensemble Learning

Demo Ensemble Learning trên bài toán Customer Churn.

## Project structure

```text
main/
├── data/
│   └── churn.csv
├── Dockerfile
├── ensemble_model.pkl
├── main.py
├── predict.py
├── requirements.txt
└── train.py
```

## Ensemble methods

Project so sánh:

1. Decision Tree
2. Bagging
3. Random Forest
4. AdaBoost
5. Gradient Boosting
6. Voting
7. Stacking

## 1. Chạy trên Google Colab

Upload/cloning repository vào Colab rồi:

```python
!pip install -r requirements.txt
!python train.py
```

Sau khi train:

```python
!python predict.py
```

## 2. Chạy local

```bash
pip install -r requirements.txt
python train.py
python predict.py
```

## 3. API

Train model trước:

```bash
python train.py
```

Sau đó:

```bash
python main.py
```

API:

```text
POST http://localhost:5000/predict
```

JSON example:

```json
{
  "Age": 45,
  "Balance": 150000,
  "Products": 1,
  "Active": 0,
  "Tenure": 3,
  "CreditScore": 600,
  "EstimatedSalary": 80000
}
```

Response:

```json
{
  "model": "Voting",
  "prediction": "Churn",
  "churn_probability": 0.72
}
```

## 4. Docker

```bash
docker build -t churn-ensemble .
docker run -p 5000:5000 churn-ensemble
```

## Dataset note

The available source for the original demo was a screenshot containing only 5 example customers. That is not enough to train a meaningful ML model.

Therefore `data/churn.csv` is synthetic data generated to follow the same churn logic. Replace it with the real dataset when available.

The metrics are for learning/demo purposes, not real banking performance.
