FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/serving/ src/serving/
COPY src/models/ src/models/
COPY src/data/dataset.py src/data/dataset.py
COPY src/data/augmentation.py src/data/augmentation.py
COPY src/data/__init__.py src/data/__init__.py
COPY src/__init__.py src/__init__.py
COPY checkpoints/ checkpoints/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
