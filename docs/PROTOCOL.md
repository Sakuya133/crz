# Protokol eksperimen empiris

## Research questions dan batas klaim

RQ1: bagaimana sumber AP/PA berkaitan dengan ranking skor dan keputusan deteksi pada
target tetap? RQ2: apakah pola bertahan lintas backbone/seed? RQ3: apakah mixed
GroupDRO memperbaiki worst-view dibanding mixed ERM dengan akses data identik?
Hipotesis terbuka; delta dapat positif, negatif, atau belum terpisahkan dari
ketidakpastian. View bersifat observasional: scanner, pasien, penyakit dan faktor
akuisisi dapat berasosiasi; tidak mengidentifikasi sebab-akibat. Tidak ada klaim
penelitian pertama, algoritme baru, atau SOTA dari penambahan kode.

## Data dan seed

Pilot 2.000 pasangan images_001: per view AP/PA 700 train, 100 validation, 200 test.
Assignment pasien **sebelum sampling** pada audit asal; diimpor lengkap, bukan
diulang dari subset file lab. Pasien yang memiliki AP dan PA tetap satu split.
Official-test patients tidak masuk train/validation. Kuota kurang, duplikat,
metadata/view tidak cocok, file hilang atau checksum salah menyebabkan error.
Pilot hanya PNG 8-bit mode L tanpa konversi. Full mencatat exclusion ilmiah terpisah
dari file belum tersedia. Dataset bukan diagnosis classifier.

Seed split/sampling/embedding/crop/training awal 1337, terpisah eksplisit. CNN
2026/42 hanya mengubah training seed. Ekstraksi SRM deterministik dengan data/crop
sama. LinearSVC random_state 1337 mengunci urutan solver; tidak diulang tiga kali
dan diklaim sebagai tiga CNN seed. Perbandingan otomatis SVM versus CNN memakai
seed 1337 yang sama. Stabilitas solver, bila diperlukan, eksperimen terpisah.

## Embedding/preprocessing

Ordinary LSB Matching memilih floor(αHW) posisi unik, bit acak, ±1 acak bila parity
tidak cocok. Piksel 0 dipaksa +1, 255 dipaksa −1; sementara signed integer mencegah
overflow uint8. α=0.2 bpp bukan 20% piksel berubah; ekspektasi sekitar 10%, jumlah
aktual dicatat. Ini **bukan LSB Matching Revisited**.

Embedding full-image **sebelum** crop 256×256. Koordinat identik cover/stego dari
hash(crop_seed,pair_id,patch_index). Tidak ada resize, JPEG, augmentation, denoise,
atau adaptive test normalization. Highpass/SRNet menerima float32 uint8/255; SRM
menerima uint8 dari crop sama. Patient ID, filename, view dan diagnosis hanya untuk
audit/sampling, bukan fitur. Inferensi menerima satu gambar tanpa cover pasangannya
sebagai informasi tambahan. Pair ID memuat payload: antar-payload 0.1/0.2/0.4 crop
dapat berbeda; antarmodel dalam payload tetap sama. Ablation lintas-payload yang
menghendaki crop identik memerlukan protokol baru, bukan perubahan diam-diam.

## A — source-specific

| Training | Validation pemilihan model/threshold | Test |
|---|---|---|
| AP | AP | AP dan PA |
| PA | PA | PA dan AP |

Target AP: AP→AP versus PA→AP. Target PA: PA→PA versus AP→PA.
Delta = **mismatched − matched**. AP→AP versus AP→PA bukan fixed-target contrast.

CNN: AdamW lr=.001, weight_decay=.0001, paired batch 8 (16 gambar), 10 epoch;
ReduceLROnPlateau memakai validation loss; checkpoint AUC source-validation
tertinggi, tie mempertahankan checkpoint terdahulu. Budget 10 epoch bukan klaim
konvergensi atau reproduksi recipe resmi SRNet. Full SRM: 34.671 fitur,
StandardScaler fit train saja; LinearSVC dual=True, max_iter=20.000, tol=1e-4.
C={.01,.1,1}, AUC source-validation tertinggi, C terkecil jika tie. Convergence
warning adalah error; tidak mengganti classifier/fit subset diam-diam.

Threshold memaksimalkan BA/Youden J pada source validation; threshold terbesar
menang jika tie. Satu threshold dibekukan untuk kedua target. Test tidak memilih
checkpoint/C/crop/normalisasi/threshold. Sensitivity=TPR stego, specificity=TNR
cover, FPR=1−specificity. AUC mengukur ranking; BA/FPR/TPR mengukur operating point
tetap. Pola AUC stabil dengan BA berubah bisa konsisten dengan perpindahan operating
point, bukan bukti kausal; tidak memperbaiki angka dengan target-test calibration.

## B — mixed-view mitigasi, opt-in

HighPassResidualCNN sama. 350 AP+350 PA training pairs dari pasien terkunci;
mixed validation identik. Balanced AP/PA batches, pasangan cover/stego utuh,
batch terakhir tidak dibuang. Inisialisasi, seed, regularisasi, budget update,
epoch, dan checkpoint macro validation AUC sama. Scheduler memakai validation
yang sama, tetapi lintasan lr bisa berbeda sebagai akibat objective.

L_g = mean classification loss untuk view g;
log q_g ← log q_g + η L_g (detach), normalisasi logsumexp;
L_robust = Σ q_g L_g; η=.01, q awal seragam. q dan counter disimpan untuk resume.
Tidak memakai BTL/generalization adjustment. AdamW weight decay sama dengan ERM.
Ini adaptasi GroupDRO existing; regularisasi penting dan tidak menjamin superiority.

Threshold tunggal dari mixed validation, tetap untuk test AP/PA. Laporkan tiap view,
macro dan worst-view (minimum AUC/BA/TPR/TNR; maksimum FPR). Gap mengecil karena view
baik memburuk saja bukan perbaikan. Keduanya melihat AP+PA, bukan unseen-view
generalization. Analyzer memeriksa images, initial weights, settings, epochs dan
effective updates; AMP skip yang merusak kesetaraan update menggagalkan kontrol.

## Statistik dan pelaporan

Paired patient-cluster bootstrap 10.000 final, CI percentile 95%; smoke 50 hanya
verifikasi. Align keys/crop/protocol/manifest/source/target/payload/seed. Resample
pasien dengan replacement, seluruh gambar/pasangan/multiplicity dipertahankan;
threshold tetap. Cross-view mode menolak model berbeda. Mode method eksplisit
mengizinkan margin SVM versus probabilitas CNN tanpa menggabungkan skor mentah.
Macro/worst B resample **pasien gabungan kedua view**, bukan dua bootstrap independen.

ΔAUC absolut; ΔBA(pp)=100(BA_candidate−BA_baseline);
relative BA(%)=100(BA_candidate−BA_baseline)/BA_baseline (nol → unavailable).
CI pasien untuk fitted models tetap; mean/SD antar-training-seed terpisah. Satu seed
punya SD unavailable, bukan nol. Jangan memperlakukan tiga seed sebagai pasien
independen tambahan. CI melintasi nol tidak membuktikan kesetaraan. Interaksi
selisih efek AP versus PA belum diuji; asimetri hanya deskriptif, bukan ditentukan
dari satu CI mencakup nol dan yang lain tidak.

Grafik/tabel berasal dari prediksi/log aktual. Synthetic dan NIH ditolak bila
dicampur satu report root. Historical pilot tidak disalin menjadi hasil paket baru.

## Pelacakan klaim dan kebaruan

| Klaim/kandidat | Eksperimen | Dukungan artefak | Status | Keterbatasan |
|---|---|---|---|---|
| Pipeline menjaga pairing/split/threshold | Synthetic E2E dan regression | reports/verification; smoke outputs | Diverifikasi software | Bukan bukti performa NIH |
| Sumber AP/PA berkaitan dengan performa | A, fixed-target paired comparisons | metrics.csv, ROC, scenario matrices, delta CI | Menunggu NIH lab | View tidak dirandomisasi; bukan sebab-akibat |
| Pola bertahan lintas model/seed | A, 3 baseline dan CNN seeds 1337/2026/42 | per-seed metrics, training_seed_variation.csv | Menunggu NIH lab | SVM solver fixed seed; bukan tiga run CNN |
| Ranking versus operating point berbeda | AUC serta BA/TPR/FPR dengan threshold source val | ROC dan metrics/deltas | Menunggu NIH lab | Tidak tuning threshold target; tidak membuktikan mekanisme |
| GroupDRO memperbaiki worst view | B, objective-only controlled ablation | mitigation.csv, control receipts, joint patient CI | Software siap; NIH pending | Keduanya melihat AP+PA; gap mengecil saja tidak cukup |
| Temuan bertahan pada pasien tambahan | Full confirmatory cohort | protocol lock, unexposed-cohort predictions | Belum dijalankan | Pasien pilot telah terpapar; freeze bukan bukti prior non-exposure |

Tidak ada algoritme baru diklaim. SRNet, SRM dan GroupDRO adalah existing methods;
adapter/runner baru bukan novelty ilmiah. Kandidat kontribusi empiris: protokol
pasien-disjoint AP/PA yang auditable, perbandingan fixed-target, dekomposisi
ranking/operating-point, dan evaluasi mitigasi terkontrol. Perlu penelusuran prior
art lebih luas serta hasil NIH sebelum klaim gap spesifik disahkan mentor.

Hipotesis untuk diskusi, bukan modul tambahan: perubahan distribusi residual
akuisisi dapat menggeser operating point lebih kuat daripada ranking. Uji dahulu
pola AUC versus BA/FPR pada target tetap; kontrol penyakit/pasien/scanner yang
tersedia secara deskriptif. Jika mitigasi gagal, hasil tersebut tetap dilaporkan.

Pola penyajian mentor dan status paper REYHealth dijelaskan pada
[METHODS_AND_REFERENCES.md](METHODS_AND_REFERENCES.md); bukan detektor pembanding.
