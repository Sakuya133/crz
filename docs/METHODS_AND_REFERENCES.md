# Metode, sumber implementasi, dan lisensi

Bibliografi: `references.bib`. DOI dipertahankan dari metadata terverifikasi pada
audit asal, bukan diciptakan dari judul. Implementasi yang dicantumkan bukan
otomatis reproduksi seluruh metode paper. Status eksperimen paket ini ada pada
receipt smoke/regression; tidak mewarisi klaim NIH dari proyek asal.

| Metode/komponen | Paper | Implementasi dan perubahan | Status/lisensi/batas |
|---|---|---|---|
| NIH ChestX-ray14 | Wang et al. 2017, [10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369) | Metadata Image Index/Patient ID/View Position, official lists; classifier cover/stego, bukan penyakit | Data tidak didistribusikan; pengguna mengikuti ketentuan [sumber NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737) |
| Ordinary LSB Matching | Ker 2005, [10.1109/LSP.2005.847889](https://doi.org/10.1109/LSP.2005.847889) | Implementasi proyek: posisi unik, random bit, ±1 parity mismatch, signed intermediate, batas 0/255 | Custom implementasi ordinary LSBM; bukan LSBMR |
| HighPassResidualCNN | Custom; GroupNorm Wu & He 2018 [10.1007/978-3-030-01261-8_1](https://doi.org/10.1007/978-3-030-01261-8_1) hanya komponen normalisasi | `models/residual_cnn.py` dipertahankan; frozen empat high-pass, residual/GN/SiLU, GAP, binary logit | Bukan SRNet atau full SRM; source milik proyek, tidak ada lisensi umum asal yang diinventarisasi |
| SRNet | Boroumand, Chen & Fridrich 2019, [10.1109/TIFS.2018.2871749](https://doi.org/10.1109/TIFS.2018.2871749), [author PDF](https://ws.binghamton.edu/fridrich/Research/SRNet.pdf) | `models/srnet.py` adaptasi PyTorch audited: 2 type-1, 5 type-2, 4 type-3, type-4, GAP, dua logits; /255, AdamW, NIH crop, recipe proyek | Adaptasi, **bukan reproduksi resmi**. Tidak menyalin source author; initializer/budget/recipe tidak sama paper |
| Full SRM + StandardScaler + LinearSVC | Fridrich & Kodovský 2012, [10.1109/TIFS.2012.2190402](https://doi.org/10.1109/TIFS.2012.2190402) | `sealwatch==2025.9`, 34.671 fitur/schema terkunci; scaler training-only, C dari source-validation, fixed decision margin | [Sealwatch](https://github.com/uibk-uncover/sealwatch) dependency [MPL-2.0](https://github.com/uibk-uncover/sealwatch/blob/main/LICENSE); sklearn BSD-3-Clause. Bukan FLD ensemble asli; margin bukan probabilitas |
| Mixed ERM versus GroupDRO | Sagawa et al., ICLR 2020, [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) | Adaptasi unadjusted exponential group weights dalam log-space; AP/PA groups, balanced paired batches, η=.01, same AdamW; no BTL/adjustment | [Author repo](https://github.com/kohpangwei/group_DRO), commit `cbbc1c5b06844e46b87e264326b56056d2a437d1`, MIT; implemented equation adaptation, not vendored repository. Existing algorithm, not novelty |
| Cover-source mismatch framing | Mallet et al. 2024, [10.1186/s13635-024-00171-6](https://doi.org/10.1186/s13635-024-00171-6) | Framing related work; empat skenario/fixed-target/bootstrap adalah desain eksperimen proyek | Referensi tidak membuktikan gap AP/PA baru atau mismatch pasti merugikan |
| HSMNet (opsional belum siap default) | Yang et al., Information Sciences 728 (2026), [10.1016/j.ins.2025.122824](https://doi.org/10.1016/j.ins.2025.122824) | Adapter audited dari proyek lama; [author source](https://github.com/diqi12/HSMNet), commit `2c9f57be9c24753308931ee7e016c9cb2958813d`; 3 hash source/kernel diverifikasi adapter | Source author **tidak disertakan**; tidak ada LICENSE pada commit yang diperiksa. Full paper belum diverifikasi menyeluruh. Dikeluarkan dari runner wajib, bukan placeholder completed |
| REYHealth (style/related embedding work) | Thandya et al. 2026, [10.1016/j.rineng.2025.108915](https://doi.org/10.1016/j.rineng.2025.108915) | Tidak diimplementasikan; bukan detector comparator | PDF tidak tersedia di workspace asal. Bagian 3/4.1–4.4, Table 7, Figures 4–7 belum dibaca; tidak mengaku memverifikasinya |

LSB Matching Revisited (Mielikäinen 2006, 10.1109/LSP.2006.870357), MI-STEG dan
paper klasifikasi penyakit di bibliografi adalah related work, bukan metode
terimplementasi. HSMNet memakai bank conv awal 30 filter yang trainable, bukan
fitur full SRM 34.671. Jangan menukar keduanya berdasarkan nama SRM.

## Distribusi

Kode proyek lama berasal dari workspace pengguna; tidak ditemukan LICENSE umum
yang memberi dasar untuk menetapkan lisensi baru atas seluruh proyek. Paket ini
tidak secara sepihak mengubah lisensinya. Sebelum publikasi source secara open
source, pemilik menentukan lisensi untuk bagian miliknya. Tidak ada source author
HSMNet disalin. Dependency di-install dari distribusinya, beserta lisensi masing-
masing; referensi paper bukan izin untuk menyalin source.

PyTorch BSD-style, torchvision BSD-3-Clause, NumPy/pandas/scipy/sklearn BSD-style,
matplotlib PSF-compatible, PyYAML/psutil MIT, Pillow HPND; periksa LICENSE distribusi
versi aktual saat redistribution. Tidak ada binary/dependency vendored dalam Git.
Source snapshot run hanya kode/config lokal untuk audit, bukan environment/data.

## Pelaporan mengikuti pola riset, bukan menyalin paper mentor

Masalah → research questions → metode/persamaan → setup → tabel/grafik aktual →
diskusi biaya/keterbatasan → kesimpulan bersyarat. Tidak menyalin teks, gambar,
angka, mekanisme embedding REYHealth, atau menggunakannya sebagai detector.
PSNR/SSIM/histogram bukan bukti tidak terdeteksi; metrik utama detector adalah
AUC/BA/TPR/TNR/FPR. Angka lintas-paper dengan dataset/protokol berbeda bukan
peningkatan langsung.
