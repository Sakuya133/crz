# CXR Steganalysis

Eksperimen NIH chest X-ray: deteksi **cover=0 / stego=1**, train AP atau PA,
test pada kedua view. Model: HighPassResidualCNN custom, SRNet adaptasi, dan
Full SRM (34.671 fitur) + StandardScaler + LinearSVC.

## Kebutuhan

Gunakan environment lab yang sudah ada: Python 3.12 dan PyTorch/torchvision
yang kompatibel dengan GPU. Tidak ada pembuatan environment atau instalasi otomatis.
Jika perlu, pasang dependensi sendiri (bukan pada environment proyek lama):

```bash
python -m pip install -r requirements.txt
```

PyTorch GPU mengikuti [petunjuk resmi](https://pytorch.org/get-started/locally/).
Command berikut dijalankan dari root repository.

## Data dan pilot

Siapkan sendiri PNG `images_001`, metadata dan official lists penuh:

```text
data/raw/<subfolder>/<image_id>.png
data/metadata/Data_Entry_2017_v2020.csv
data/metadata/train_val_list.txt
data/metadata/test_list.txt
private-transfer/pilot_audited/{bundle.json,cover_stego.csv,patient_assignments.csv}
```

Bundle pilot sudah tersedia lokal; **bawa terpisah, tidak ikut GitHub**.
Untuk data di disk lain, isi `configs/local.yaml` dari `local.example.yaml`,
lalu tambahkan `--local configs/local.yaml` pada command data/doctor/lab.

```bash
# Impor split terkunci + buat stego; raw tidak ditimpa.
python scripts/data.py import-pilot --bundle private-transfer/pilot_audited --generate-stego
python scripts/doctor.py --config configs/pilot_existing.yaml --data --require-cuda --require-srm

# Satu baseline: train AP/PA, evaluate empat skenario, bootstrap, grafik/tabel.
python scripts/lab.py --dry-run --models highpass --seeds 1337 --output outputs/pilot_highpass
python -u scripts/lab.py --execute --models highpass --seeds 1337 --output outputs/pilot_highpass
```

Buka **`outputs/pilot_highpass/report/README.md`**.
Untuk ketiga model, ganti dengan `--models highpass srnet srm_svm` dan output baru.
Untuk melanjutkan run terputus, ulang command execute dengan `--resume`.

```bash
python scripts/lab.py --status --output outputs/pilot_highpass
python scripts/report.py --run-root outputs/pilot_highpass --output reports/pilot_highpass
```

Pilot tetap 2.000 pasangan, LSB Matching 0,2 bpp, embedding sebelum paired crop,
patient-disjoint, checksum diperiksa, threshold hanya dari source validation.
Test tidak untuk tuning. Runner serial; tidak mengganti training NIH ke CPU.

## Opsi

- ERM/GroupDRO terpisah: `--config configs/pilot_mixed.yaml`, output baru.
- Full NIH: `configs/full_all_eligible.yaml`; [command full](docs/FULL_NIH_RUNBOOK.md).
- [Protokol](docs/PROTOCOL.md) · [Referensi metode](docs/METHODS_AND_REFERENCES.md).
- Smoke sintetis: `python scripts/smoke.py --output outputs/smoke_lab` (output baru).
- Entry point tetap di `scripts/`, modul di `src/`, preset di `configs/`, tes di `tests/`.
  Setiap script memiliki `--help`. Tidak ada model PENet yang ditambahkan.

Dataset, manifest privat, checkpoint, prediksi, dan hasil lokal tidak ikut Git.
Kode/smoke siap bukan berarti eksperimen NIH selesai.
