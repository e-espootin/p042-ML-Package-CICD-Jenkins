# ML Package dev for titanic prediction

## dev & learn
- [x] setup env
- [x] push ML module to Repo
- [x] setup PAT
- [x] setup Jenkins polling
- [ ] setup Jenkins pipeline

...

# setup env
```python
conda create -n jenkins-env python=3.12
conda activate jenkins-env
pip install -r requirements.txt

# install package locally
pip install src/
```

## sample data for test in postman
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
docker run -d -it --name titanic_pred -p 8005:8005 titanic_pred:v1

# push to my docker hub
docker tag titanic_pred:v1 e4espootin/titanic_pred:v1
docker push e4espootin/titanic_pred:v1

# run docker container
docker run -d -it --name titanic_pred -p 8005:8005 e4espootin/titanic_pred:v1

# run model
docker exec titanic_pred python prediction_model/training_pipeline.py
docker exec titanic_pred pytest -v --junitxml TestResults.xml --cache-clear
docker cp titanic_pred:/code/src/TestResults.xml TestResults.xml
```
