# a

```python
conda create -n jenkins-env python=3.12
conda activate jenkins-env
pip install -r requirements.txt

# install package locally
pip install src/
```

```json
{
  "Pclass": 1,
  "Sex": "Female",
  "Age": 38,
  "SibSp": 1,
  "Parch": 0,
  "Cabin": "C85",
  "Fare": 71,
  "Embarked": "C"
}
result > survived

{
  "Pclass": 1,
  "Sex": "Male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 1,
  "Cabin": "S1",
  "Fare": 1,
  "Embarked": "S"
}
result > not survived
```

# docker commands
```bash
docker build -t titanic_pred:v1 .

docker run -p 5000:5000 titanic_pred:v1
```