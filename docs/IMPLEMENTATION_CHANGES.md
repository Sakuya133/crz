# Apa yang diambil dan diubah

Sumber: proyek `cxr-steganalysis` yang sudah diaudit. Git sumber bukan repository
valid pada saat ekspor; tidak mengarang commit. Hash berkas asal/hasil disimpan
dalam SOURCE_PROVENANCE.json. Raw data, venv, checkpoint, hasil pilot dan source
author HSMNet tidak disalin. Ekspor bundle pilot hanya CSV privat+checksum.

| Area | Dipertahankan | Tambahan/perubahan paket lab |
|---|---|---|
| Model | HighPassResidualCNN dan SRNet: arsitektur/forward sama, hanya whitespace akhir berkas berbeda; identitas/checkpoint version sama | Tidak mengganti arsitektur; HSM adapter tetap guarded, tidak diantrekan |
| Embedding | Ordinary LSBM, seed per-image, full-image sebelum crop | Guard menolak overwrite PNG; importer mengecek regenerasi terhadap checksum pilot |
| Data | Metadata join, global patient split, paired dataset/crop, full enrollment locked assignments | Private transport/export/import, pin hash pilot, path-only local overlay, relokasi audited, readiness receipt/checksum ulang file berubah |
| Training | Optimizer/scheduler/AMP/RNG/loader/GroupDRO resume yang diuji | CLI menolak direktori existing tanpa resume dan CPU fallback implisit; log/plan baru terpisah |
| SRM | Full 34.671, train-only scaler, LinearSVC C source validation, chunked cache | Solver seed override juga dicatat konsisten di config; queue satu solver seed saja |
| Evaluasi | Source-validation threshold fixed, per-sample score, model.eval, protocol lock | File artefak ditemukan relatif terhadap folder run agar report bisa dipindahkan |
| Statistik | Paired patient-cluster multiplicity, mode cross-view/method terpisah, mixed joint bootstrap | Tambah FPR CI; analyzer generic multi-seed tanpa pooling pasien; kontrol objective-only B |
| Eksekusi | Low-level CLI modular | lab.py --dry-run/--execute/--resume/--status, lock, exit status, time budget, sampled tree RSS, immutable configs/artifacts |
| Full NIH | Semua eligible, exposed versus additional official-test patients | Expected 112.120 metadata records, full subset guard, preflight/prepare wrapper, payload presets terpisah |
| Laporan | Metrik rekonstruksi dari CSV | Report generic A/B/multi-seed, ROC/matrix/CI/curves, CSV/LaTeX/PNG/PDF, Indonesia, explicit synthetic labels |

Kode baru tidak mengimpor proyek asal, menggunakan symlink ke sana, atau membawa
path absolut asal dalam preset/implementation. Satu path asal pada dokumentasi
adalah command **ekspor sekali**; tidak diperlukan saat runtime lab.

Bagian yang belum dibuktikan: performa NIH di lab, konvergensi 10 epoch, konsistensi
multi-seed, efektivitas mitigasi, interaksi asimetri AP versus PA, resource full
LinearSVC aktual, compatibility CUDA/driver lab, dan full HSMNet reproduction.
