# Status verifikasi paket — 27 September 2026

**Kode siap + synthetic smoke lulus; bukan eksperimen NIH selesai.**

- Regression: **77 passed, 1 skipped**, 0 failures/errors; [JUnit](regression.xml).
  Yang dilewati adalah opt-in CUDA resume test. Venv baru memakai build CPU.
- Smoke end-to-end: **78.05 detik**, 24 pasangan/12 pasien sintetis, kedua view per
  pasien, seed 1337, crop32, CPU. Ketiga baseline wajib, empat skenario per baseline,
  mixed ERM/GroupDRO, source-validation threshold, cache SRM 34.671, resume queue,
  bootstrap 50 dan laporan lulus. Bootstrap 50 bukan CI final penelitian.
- [25 CLI help/import checks](cli_help.json) lulus; compileall dan shell syntax lulus.
- [Environment](environment.json): venv baru Python3.12.3, torch2.13.0+cpu,
  sealwatch2025.9; `pip check` bersih. GPU fisik terlihat oleh driver, tetapi
  **build CPU ini tidak menyediakan CUDA**; lab harus memilih pasangan Torch/GPU.
- [Probe encoder NIH](nih_encoder_probe.json): 3 gambar pilot diregenerasi in-memory,
  SHA-256 PNG cocok dengan manifest asal. Tidak ada training, tidak menulis raw/stego,
  bukan audit ulang penuh 2.000 pasangan atau hasil performa NIH baru.
- Synthetic full-extension fixture memeriksa locked patients, official-test baru,
  exposed-versus-confirmatory, streaming prepare, dan penolakan subset tersembunyi.
- Unit tests memeriksa scaler hanya fit training, test tidak memengaruhi fit SVM,
  alignment/cluster multiplicity, threshold validation-only, RNG/optimizer/q resume,
  no overwrite, duplicate basename, checksum, local config, dan queue failure status.

## Artefak lokal yang dapat dibuka

- [Laporan A sintetis](../synthetic_verification_final/A/README.md)
  dan [caption/figure index](../synthetic_verification_final/A/figures/README.md).
- [Laporan B sintetis](../synthetic_verification_final/B/README.md)
  dan [ringkasan Indonesia](../synthetic_verification_final/B/mentor_progress_id.md).
- Full run/log/checkpoint synthetic: `outputs/synthetic_smoke_final_20260927/`.
- [Receipt ringkas](summary.json), [asal/perubahan kode](../../docs/IMPLEMENTATION_CHANGES.md).

Artefak besar/synthetic figures diabaikan Git. Pada clone baru, regenerasikan
dengan `python scripts/smoke.py --output outputs/smoke_lab`; report otomatis ada
di `outputs/smoke_lab/A/report` dan `B/report`. Jangan menafsirkan angka acak smoke
sebagai deteksi NIH. PNG/PDF learning curve dan delta CI telah diperiksa render;
label SYNTHETIC jelas, label/legend/error interval tidak terpotong.

## Perlindungan sumber dan status pekerjaan

70 berkas sumber yang dipakai ulang diperiksa tetap tidak berubah sejak ekspor.
Manifest pilot dan checkpoint historical AP/PA cocok hash awal; lihat summary.json.
Proyek asal tidak di-reset, raw/hasil/checkpoint lama tidak dipindahkan/disalin.
Pilot lama telah dihentikan. Tidak ada job training berjalan pada akhir penyiapan.

Repo baru diinisialisasi lokal untuk pengecekan ignore, **tanpa commit, remote atau
push**. `.venv`, data, private bundle, cache, checkpoint, konfigurasi lokal dan
prediction rows ter-ignore. Hanya petunjuk data ikut source. Lisensi keseluruhan
kode proyek perlu ditentukan pemilik sebelum memberi grant open-source; source
author HSMNet tidak disertakan.

Belum dibuktikan: training/konvergensi NIH di lab, full-dataset throughput/peak solver,
efek mismatch atau mitigasi, kestabilan lintas seed pada NIH, asimetri signifikan,
dan CUDA stack lab. Full NIH membutuhkan data lengkap serta resource memadai.
