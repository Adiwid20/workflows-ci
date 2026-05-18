FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.5.2 \
    imbalanced-learn==0.12.4 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    cloudpickle==3.1.2 \
    psutil==7.0.0 \
    pyarrow==18.1.0 \
    scipy==1.14.1


ARG MODEL_PATH
COPY ${MODEL_PATH} /app/model


EXPOSE 8080


CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]
