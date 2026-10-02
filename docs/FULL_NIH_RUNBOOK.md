# NIH lebih besar / full all eligible — dijalankan di lab

Tidak dijalankan saat penyiapan paket ini. Mulai dari payload .2, seed 1337;
setelah protokol/budget diputuskan melalui validation, gunakan 2026 dan 42.
Payload .1/.4 adalah perluasan terpisah. Jangan meluncurkan seluruh matriks tanpa
profiling. `full_all_eligible` berarti semua NIH yang memenuhi kriteria, bukan
subset seimbang atau hanya file yang kebetulan ada.

Preset perluasan: `configs/full_bpp01.yaml` dan `configs/full_bpp04.yaml`.
Jika memakai full.local.yaml yang meng-override pair_manifest/generation/stego,
buat overlay payload yang sesuai juga; jangan mengarahkan .1/.4 ke output bpp02.
Binding prepare menolak penggunaan direktori payload yang tidak cocok.

## Data dan ruang

Metadata NIH penuh (112.120 image records), kedua official lists, dan PNG dari
seluruh bagian `images_001` sampai `images_012` disiapkan pengguna. Arsip tidak
perlu berada di proyek; tidak di-download/diekstrak pipeline. Inventory recursive
PNG adalah sumber kelengkapan, bukan sekadar nama arsip. Semua missing/unreadable
harus diselesaikan sebelum prepare full. Mode selain L, view bukan AP/PA, dan
gambar lebih kecil dari crop adalah exclusion ilmiah yang dicatat tersendiri.

Impor pilot dahulu, agar seluruh assignment pasien terkunci dipertahankan.
Pasien tambahan official-test tetap test; official-train tambahan dialokasikan
validation dengan deterministic patient hash 1/8, sisanya train. Gambar tambahan
dari pasien pilot mengikuti split pasiennya. Pasien pilot test ditandai exposed;
confirmatory hanya pasien official-test tambahan, **bukan gambar baru pasien lama**.

Default path: `data/manifests/full` inventory, `data/manifests/full/bpp02` pasangan
dan stego. Untuk disk lain, buat `configs/full.local.yaml` (ignored) dari contoh:

```yaml
paths:
  metadata: /mnt/nih/metadata/Data_Entry_2017_v2020.csv
  official_train_list: /mnt/nih/metadata/train_val_list.txt
  official_test_list: /mnt/nih/metadata/test_list.txt
  raw_dir: /mnt/nih/images
  patient_assignments: /mnt/experiments/nih_full/patient_assignments.csv
  split_manifest: /mnt/experiments/nih_full/splits.csv
  pair_manifest: /mnt/experiments/nih_full/bpp02/cover_stego.csv
  generation_manifest: /mnt/experiments/nih_full/bpp02/stego_generation.csv
  stego_dir: /mnt/experiments/nih_full/bpp02/stego
```

`pilot_patient_assignments` dan `pilot_pair_manifest` tetap menunjuk impor pilot
lokal default; override juga bila pilot disimpan di disk lain. Semua path writable
terpisah dari raw. Ketika relokasi input sesudah preflight, buat snapshot inventory
baru; jangan mengedit binding snapshot lama. Full metadata yang tidak lengkap harus
diganti file resmi penuh, bukan dipangkas agar cocok file lokal.

```bash
python scripts/full_data.py preflight --config configs/full_all_eligible.yaml \
  --local configs/full.local.yaml
# Lihat summary.json, counts.csv, missing_images.csv, inventory.csv.
# Setelah inventory lengkap dan disk cukup:
python scripts/full_data.py prepare --config configs/full_all_eligible.yaml \
  --local configs/full.local.yaml
python scripts/doctor.py --config configs/full_all_eligible.yaml \
  --local configs/full.local.yaml --data --require-cuda
```

Preflight tidak overwrite snapshot lama; jika incomplete, pertahankan diagnosisnya
dan arahkan split/assignment ke folder snapshot baru setelah melengkapi gambar.
Prepare streaming satu gambar penuh, record atomic per gambar; command sama dapat
diulang untuk resume setelah memverifikasi checksum record. Tidak memuat semua
gambar ke RAM. Tidak ada lazy embedding/crop yang belum dibuktikan ekuivalen.

## Kontrol ukuran (analisis utama), lock, training, evaluasi

Analisis **utama** = `configs/full_size_matched.yaml`: subset baris dari manifest full
(file cover/stego sama, tidak disalin). Per split train/validation diambil
min(AP, PA) pasangan per view, urutan `stable_seed(seeds.sampling=20261002, pair_id,
split)`; assignment pasien tidak berubah; test tidak dikurangi. Subset sama untuk semua
model dan training seed. `full_all_eligible` = analisis **tambahan**. Pasangan setara
tidak berarti pasien setara (proyeksi train: AP 6.797 vs PA 15.191 pasien) — laporkan
sebagai keterbatasan; lihat `counts_control` di `size_matched/size_control.json`.

```bash
python scripts/full_data.py size-control --config configs/full_size_matched.yaml
python scripts/doctor.py --config configs/full_size_matched.yaml --data --require-cuda
# SATU lock untuk seluruh rencana bpp0.2: kedua preset, highpass+srnet, AP/PA,
# training seed 1337/2026/42, hyperparameter, metrik, perbandingan. Sekali saja.
python scripts/freeze_protocol.py --plan configs/full_size_matched.yaml configs/full_all_eligible.yaml
```

`lab.py --execute` untuk preset full menolak mulai bila lock belum ada atau run tidak
tercantum di dalamnya; evaluator confirmatory memeriksa lock yang sama. Run boleh
bertahap per seed, tetapi semuanya harus anggota lock tersebut. Perubahan resep
setelah lock = rencana baru dengan lock/output baru, bukan menimpa.

```bash
python scripts/lab.py --dry-run --config configs/full_size_matched.yaml --seeds 1337 \
  --output outputs/full_size_matched_seed1337
python -u scripts/lab.py --execute --config configs/full_size_matched.yaml --seeds 1337 \
  --output outputs/full_size_matched_seed1337 --budget-hours 8
# terputus/budget habis: ulangi dengan --resume. Lalu --seeds 2026 dan 42 (output baru),
# kemudian configs/full_all_eligible.yaml dengan pola yang sama (outputs/full_bpp02_seed*).
python scripts/report.py --run-root outputs/full_size_matched_seed1337 \
  --output reports/full_size_matched_seed1337
```

Hanya `confirmatory_unseen_patient` dievaluasi. Kohort ini = pasien official-test yang
tidak punya gambar terevaluasi di pilot (58 pasien pilot test = `pilot_exposed`),
termasuk 4 pasien terkunci test di pilot yang tidak terpilih kuota (661, 849, 886,
1330); diputuskan tetap confirmatory: tidak dipakai train/validation, tidak pernah
dievaluasi, dan tidak memengaruhi pengembangan model. Lock bukan bukti bahwa manusia tidak pernah melihat pasien itu. Exposed
cohort, bila dianalisis, ke report root terpisah sebagai exploratory.

## Perhitungan sumber daya

Formula untuk N cover, satu crop: cache SRM float32 = N×2×34.671×4 byte.
2.000 cover → 0.517 GiB; 112.120 cover → 28.96 GiB sebelum indeks/provenance.
Solver memakai float64: guard konservatif (5 n_train + 2 n_val)×34.671×8 byte,
dengan n menghitung gambar cover+stego source view. Ditolak bila >70% RAM tersedia.
Memmap tidak menghilangkan advanced-index/scaler/liblinear copies. Full SRM tidak
otomatis dimasukkan ke default full queue dan tidak diganti classifier/fit subset.

Sebagai kapasitas awal, metadata penuh pada audit asal memberi upper allocation
sekitar AP 72 GiB / PA 134 GiB (sekitar 103/192 GiB **RAM available** agar guard
70% lolos). Eligibility aktual lab dapat mengubah angka; gunakan resource_estimates
preflight/doctor, dan ukur peak solver sebelum menyatakan resource cukup.

Stego mode-L 1024²: 2.000 cover tak terkompresi ≈1.95 GiB; 112.120 ≈109.49 GiB per
payload; prepare memakai safety margin 10% ≈120.44 GiB free untuk jumlah penuh.
PNG bisa lebih kecil tetapi tidak diasumsikan. Sediakan raw+stego+cache+checkpoint
dan output tambahan; jangan menggandakan tiga payload sebelum mengukur disk.

Audit asal, bukan benchmark lab: RTX4060 Laptop 8 GB, crop256 batch8, SRNet 10 epoch
sekitar 168–172 detik/source (loop epoch), peak allocated VRAM ≈1.49 GiB. Ekstraksi
SRM 2.000 pasangan ≈60 menit dengan kombinasi 4→8 workers; bagian 8 worker ≈.685
pasangan/detik → ekstrapolasi 112.120 ≈45.5 jam **ekstraksi saja**, bukan janji lab
atau waktu solver. Hasil pengukuran baru dicatat pada queue.json/resources.csv.
Default 2 workers lebih konservatif dan mungkin lebih lambat. Pengukuran synthetic
crop32 tidak boleh dipakai untuk memprediksi throughput NIH crop256/1024.

Tetapkan budget/recipe sebelum konfirmatori. Tidak ada mixed full implicit balanced
sampling: Protokol B 350+350 hanya preset pilot; perluasan B memerlukan preset
ukuran eksplisit, jangan menyebutnya full all eligible.
