FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY . /code

# set permissions
RUN chmod +x /code

RUN pip install --no-cache-dir --upgrade -r /code/src/requirements.txt

EXPOSE 8005

WORKDIR /code

ENV pythonpath "${PYTHONPATH}:/code"

# CMD pip install -e ./src 
# CMD ["python", "prediction_model/training_pipeline.py"]
# WORKDIR /code
# CMD ["python","main.py"]

RUN chmod +x start.sh
CMD ["./start.sh"]