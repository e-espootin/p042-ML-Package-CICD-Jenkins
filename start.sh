#!/bin/bash
pwd
# install package
pip install -e ./src
echo 'package installed'

# run the training pipeline
python src/prediction_model/training_pipeline.py
echo 'pipeline executed'

# run the main script
python main.py