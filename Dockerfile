# Gunakan base image Python
FROM python:3.9-slim

# Tetapkan working directory di dalam container
WORKDIR /app

# Salin file requirements.txt untuk menginstal dependencies
COPY requirements.txt .

# Instal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua kode aplikasi ke dalam container
COPY . .

# Expose port 8000 (port default FastAPI)
EXPOSE 8000

# Jalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
