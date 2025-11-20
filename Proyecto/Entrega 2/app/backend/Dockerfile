FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
# Install system deps needed by some Python wheels (e.g. libgomp for LightGBM)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
