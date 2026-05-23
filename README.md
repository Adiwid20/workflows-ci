# Sistem Prediksi Persetujuan Pinjaman dengan CI/CD Pipeline

Proyek ini adalah implementasi end-to-end machine learning system untuk memprediksi persetujuan pinjaman menggunakan Random Forest Classifier. Sistem ini dilengkapi dengan automated CI/CD pipeline yang terintegrasi dengan MLflow, Docker, dan Google Drive untuk production-ready deployment.

## 📋 Deskripsi Proyek

Sistem ini dirancang untuk membantu lembaga keuangan dalam mengotomatisasi proses evaluasi permohonan pinjaman. Model machine learning yang digunakan dapat memprediksi apakah suatu permohonan pinjaman layak disetujui atau ditolak berdasarkan berbagai fitur yang tersedia dalam dataset.

### Apa yang Dilakukan Proyek Ini?

Proyek ini mengimplementasikan **automated ML pipeline** yang mencakup:

1. **Training Model Machine Learning**
   - Algoritma: Random Forest Classifier
   - Data balancing menggunakan SMOTE untuk menangani imbalanced dataset
   - Train-test split dengan stratified sampling (80:20)
   - Evaluasi lengkap dengan multiple metrics

2. **Experiment Tracking dengan MLflow**
   - Tracking semua eksperimen dan hyperparameters
   - Versioning model otomatis
   - Menyimpan artifacts (model, confusion matrix, ROC curve, precision-recall curve)
   - Logging metrics (accuracy, precision, recall, f1-score)

3. **CI/CD Automation**
   - Otomatis training setiap push ke branch main
   - Build Docker image dengan model terbaru
   - Upload artifacts ke Google Drive sebagai backup
   - Push Docker image ke Docker Hub untuk deployment

4. **Containerization & Deployment**
   - Model di-package dalam Docker container
   - REST API endpoint untuk prediksi real-time
   - Ready untuk production deployment

## 🚀 Fitur Utama

- ✅ **Automated Training Pipeline**: Training otomatis via GitHub Actions
- ✅ **Experiment Tracking**: MLflow untuk tracking dan versioning
- ✅ **Data Balancing**: SMOTE untuk handling imbalanced data
- ✅ **Cloud Backup**: Otomatis upload ke Google Drive
- ✅ **Containerized Deployment**: Docker image ready-to-deploy
- ✅ **REST API**: Model serving dengan MLflow
- ✅ **Comprehensive Evaluation**: Multiple metrics dan visualisasi

## 🛠️ Teknologi yang Digunakan

| Kategori | Teknologi |
|----------|-----------|
| **Language** | Python 3.12 |
| **ML Framework** | scikit-learn, imbalanced-learn |
| **Experiment Tracking** | MLflow 2.19.0 |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |
| **Cloud Storage** | Google Drive API |
| **Model Serving** | MLflow Model Server |

## 📁 Struktur Proyek

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── main.yml                    # CI/CD pipeline configuration
├── MLProject/
│   ├── MLProject                       # MLflow project configuration
│   ├── conda.yaml                      # Environment dependencies
│   ├── modelling.py                    # Script training model
│   ├── upload_to_gdrive.py             # Script upload ke Google Drive
│   └── loan_data_preprocessed.csv      # Dataset preprocessed
├── mlruns/                             # MLflow tracking directory
├── Dockerfile                          # Docker configuration
├── .gitignore                          # Git ignore rules
└── README.md                           # Dokumentasi proyek
```

## 🔧 Cara Menjalankan Proyek

### Prerequisites

Pastikan sudah terinstall:
- Python 3.12
- pip atau conda
- Git
- Docker (opsional, untuk deployment)

### 1. Clone Repository

```bash
git clone <repository-url>
cd Workflow-CI
```

### 2. Setup Environment

**Menggunakan pip:**
```bash
pip install mlflow==2.19.0 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 \
            imbalanced-learn==0.12.4 matplotlib==3.9.2 seaborn==0.13.2 \
            pyarrow==18.1.0 cloudpickle==3.1.2 psutil==7.0.0 scipy==1.14.1
```

**Menggunakan conda:**
```bash
conda env create -f MLProject/conda.yaml
conda activate mlflow_loan-approval-clasification
```

### 3. Menjalankan Training Lokal

**Menggunakan MLflow:**
```bash
mlflow run MLProject --env-manager=local --experiment-name "Loan_Approval"
```

**Atau jalankan script langsung:**
```bash
cd MLProject
python modelling.py
```

### 4. Melihat Hasil Eksperimen

Buka MLflow UI:
```bash
mlflow ui
```

Akses di browser: `http://localhost:5000`

## 🤖 Model Details

### Algoritma: Random Forest Classifier

**Hyperparameters:**
- `n_estimators`: 20 (jumlah decision trees)
- `max_depth`: 10 (kedalaman maksimal tree)
- `random_state`: 42 (untuk reproducibility)

### Data Preprocessing

1. **Train-Test Split**: 80% training, 20% testing
2. **Stratified Sampling**: Mempertahankan proporsi kelas
3. **SMOTE**: Oversampling untuk menangani imbalanced data pada training set

### Evaluasi Model

Model dievaluasi menggunakan:
- **Accuracy Score**: Overall correctness
- **Classification Report**: Precision, Recall, F1-Score per class
- **Confusion Matrix**: Visualisasi prediksi vs actual
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **Precision-Recall Curve**: Trade-off precision dan recall

## 🔄 CI/CD Pipeline

Pipeline otomatis berjalan setiap push atau pull request ke branch `main`:

### Workflow Steps:

1. **Checkout Repository** - Clone code dari GitHub
2. **Setup Python 3.12** - Install Python environment
3. **Install Dependencies** - Install semua library yang dibutuhkan
4. **Verify Dependencies** - Validasi versi library
5. **Create Google Credentials** - Setup credentials untuk Google Drive
6. **Run MLflow Project** - Training model dengan MLflow
7. **Find Model Path** - Locate trained model artifacts
8. **Upload to Google Drive** - Backup model dan artifacts ke cloud
9. **Setup Docker Buildx** - Prepare Docker build environment
10. **Build Docker Image** - Create container image dengan model
11. **Login to Docker Hub** - Authenticate ke Docker registry
12. **Push Docker Image** - Deploy image ke Docker Hub

### Konfigurasi GitHub Secrets

Tambahkan secrets berikut di repository settings:

| Secret Name | Deskripsi |
|-------------|-----------|
| `GDRIVE_CREDENTIALS` | Service account JSON untuk Google Drive API |
| `GDRIVE_FOLDER_ID` | ID folder Google Drive untuk menyimpan artifacts |
| `DOCKER_HUB_USERNAME` | Username Docker Hub |
| `DOCKER_HUB_ACCESS_TOKEN` | Access token Docker Hub |

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build --build-arg MODEL_PATH=<path-to-model> -t loan-approval-model:latest .
```

Contoh:
```bash
docker build --build-arg MODEL_PATH=mlruns/310588873156251133/044953eda1d241a9bb9c7a04c3162a04/artifacts/model -t loan-approval-model:latest .
```

### Run Container

```bash
docker run -p 8080:8080 loan-approval-model:latest
```

Model akan tersedia di: `http://localhost:8080`

### Melakukan Prediksi

Kirim POST request ke model endpoint:

```bash
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
      "columns": ["feature1", "feature2", "feature3", ...],
      "data": [[value1, value2, value3, ...]]
    }
  }'
```

## 📊 Monitoring dan Logging

### MLflow Tracking

Semua eksperimen tersimpan di MLflow dengan informasi:
- Run ID dan timestamp
- Hyperparameters yang digunakan
- Metrics (accuracy, precision, recall, f1-score)
- Artifacts (model file, visualisasi, confusion matrix)

### Google Drive Backup

Setiap training run otomatis di-backup ke Google Drive dengan struktur:
```
Google Drive Folder/
└── <run_id>/
    ├── artifacts/
    │   ├── model/
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── precision_recall_curve.png
    ├── metrics/
    └── params/
```

## 🐛 Troubleshooting

### Error: Model path not found
**Solusi**: Pastikan MLflow project sudah dijalankan dan model tersimpan di `mlruns/` directory.

### Error: Google Drive authentication failed
**Solusi**: 
- Periksa service account credentials sudah benar
- Pastikan service account memiliki akses ke folder Google Drive
- Verifikasi `GDRIVE_FOLDER_ID` sudah benar

### Error: Docker build failed
**Solusi**:
- Pastikan `MODEL_PATH` argument sudah benar
- Verifikasi model file ada di lokasi yang ditentukan
- Cek Docker daemon sudah running

### Error: Dependencies conflict
**Solusi**:
- Gunakan virtual environment yang bersih
- Install dependencies sesuai versi di `conda.yaml`
- Hapus cache pip: `pip cache purge`

## 📈 Workflow Lengkap

```
┌─────────────────┐
│  Push to GitHub │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ GitHub Actions      │
│ Triggered           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Install Dependencies│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Train Model         │
│ (MLflow Project)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Log to MLflow       │
│ (Metrics, Artifacts)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Upload to           │
│ Google Drive        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Build Docker Image  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Push to Docker Hub  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Ready for Deployment│
└─────────────────────┘
```

## 🤝 Kontribusi

Jika ingin berkontribusi pada proyek ini:

1. Fork repository
2. Buat branch baru (`git checkout -b feature/improvement`)
3. Commit perubahan (`git commit -am 'Add new feature'`)
4. Push ke branch (`git push origin feature/improvement`)
5. Buat Pull Request

## 📝 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan submission Dicoding - Kelas Membangun Sistem Machine Learning.
