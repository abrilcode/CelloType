# Agreement study: CelloType (Cellpose-SAM) vs a published StarDist workflow

Goal: check that CelloType's nuclear segmentation and per-nucleus marker measurements agree with
an established, published nuclei-analysis workflow, on the same images, automatically, with
quantitative metrics and timing. Tested on five datasets; all met the agreement thresholds set
before the comparison.

**Scope.** This is an agreement study between two automatic methods. The reference is another
automatic segmentation, not a manual annotation, so the results show that both methods give
concordant segmentations and readouts on these images; they do not establish the accuracy of
either method. All runs used the GPU at full resolution (no downscaling), no background
subtraction, and nuclear segmentation only.

## Method (shared by all datasets)

**Reference method, reproduced exactly** from a published ImageJ/FIJI multichannel nuclear-analysis
macro: StarDist model "Versatile (fluorescent nuclei)" (`2D_versatile_fluo`), `normalizeInput`
percentiles 1.0 to 99.8, `probThresh=0.5`, `nmsThresh=0.4`, `excludeBoundary=2` px, on the DAPI
channel; no background subtraction. Run in Python via the `stardist` package in an isolated venv.
The ImageJ plugin wraps the same TensorFlow model, so the segmentation is identical; running the
FIJI GUI headless is impractical and unnecessary.

**Our method**: `segment.segment_channel` (Cellpose-SAM `cpsam`) on the same DAPI channel, exactly
as the app runs it, **on the GPU (Apple MPS)** (the harness prints and checks the device). On top of
StarDist's own boundary exclusion, objects with any pixel within 3 px of the image border
(scikit-image `clear_border`, `buffer_size=2`) are removed from both label sets. (The app itself
now leaves out objects touching the image border by default.) Both label sets are measured with the SAME code
(`measure.measure_field`), so any difference in the readout comes only from the masks.

**What we measure (and why):**
- Segmentation agreement: foreground **Dice** + **IoU** (do the nuclear regions overlap); object
  **count** per image (+ Pearson r across images); instance **F1 at IoU>=0.5** and **mean IoU of
  matched nuclei** (are they the SAME objects one-to-one).
- Measurement agreement: **per-nucleus mean intensity** of each marker on matched nuclei -> Pearson r.
- Timing: reference (StarDist) vs ours (Cellpose) wall-clock.

Agreement thresholds, set before the comparison: Dice >= 0.85, count r >= 0.90, F1@0.5 >= 0.70,
per-marker intensity r >= 0.90.

## Summary (5 datasets, all within the thresholds; Cellpose on GPU/MPS)
| Dataset | Fields | Nuclei ref/ours | Dice | F1@IoU.5 | matched IoU | count r | intensity r (markers) | ref s / ours s |
|---|---|---|---|---|---|---|---|---|
| Public example set | 16 | 3862 / 3768 | 0.950 | 0.949 | 0.897 | 0.9994 | 0.998 / 0.998 | 8.6 / 178 |
| Maru (FN/R1 D40) | 11 | 1218 / 1233 | 0.956 | 0.965 | 0.910 | 0.9978 | 0.999 / 0.9996 | 4.6 / 121 |
| ACTH D40 | 15 | 3173 / 3151 | 0.961 | 0.962 | 0.907 | 0.9973 | 0.999 / 0.998 | 8.5 / 164 |
| IF FOXA2/NANOG D0/D7/D15 | 21 | 10729 / 10663 | 0.948 | 0.942 | 0.881 | 0.9996 | 0.997 / 0.998 | 15 / 230 |
| EpCam Dia 60 (from CZI) | 15 | 2403 / 2304 | 0.964 | 0.949 | 0.921 | 0.9914 | 0.999 / 0.998 | 12.8 / 472 |

All 5 datasets meet every threshold. Dice ~0.95, F1@IoU>=0.5 ~0.94-0.97, per-nucleus marker
intensity r ~= 0.997-0.9996. Cellpose ran on Apple MPS (GPU) in every run.

---

## Dataset 1: public example set
16 multichannel TIFF (Ch1 DAPI, Ch2 NANOG-488, Ch3 OCT6-mCherry), 8-bit, 960x1280, 4 conditions x
4 fields. Images: `20X-Clon3-Indif-minusDox-D0-Foto1..4`, `20X-Clon3-Indif-plusDox-D1-Foto1..4`,
`...-plusDox-D2-Foto1..4`, `...-plusDox-D3-Foto1..4` (16 files).

| Metric | Value | Threshold | Pass |
|---|---|---|---|
| mean foreground Dice | 0.950 | >= 0.85 | yes |
| mean instance F1 @ IoU>=0.5 | 0.949 | >= 0.70 | yes |
| mean IoU of matched nuclei | 0.897 | (report) | - |
| nucleus-count Pearson r | 0.9994 | >= 0.90 | yes |
| per-nucleus intensity r, NANOG | 0.998 | >= 0.90 | yes |
| per-nucleus intensity r, OCT6 | 0.998 | >= 0.90 | yes |

Nuclei: 3862 (reference) vs 3768 (ours). **RESULT: PASS.**
Timing: reference 8.6 s total (~0.5 s/img, CPU-TF); Cellpose-SAM 178 s (~11 s/img, MPS).

---

## Dataset 2: Maru (FN/R1 D40)
Grouped single-channel TIFF per field (DAPI + FOXA2-488 + NANOG-594), 8-bit, 960x1280; conditions
FN-D40 and R1-D40 (plus one FN-D15 DAPI-only field). Some fields are incomplete (FN-D15-Foto1 has
DAPI only; R1-D40-Foto1 has no FOXA2) and are handled gracefully.
Fields (11): `20X-FN-D15-Foto1`, `20X-FN-D40-Foto1..5`, `20X-R1-D40-Foto1..5`.

| Metric | Value | Threshold | Pass |
|---|---|---|---|
| mean foreground Dice | 0.956 | >= 0.85 | yes |
| mean instance F1 @ IoU>=0.5 | 0.965 | >= 0.70 | yes |
| mean IoU of matched nuclei | 0.910 | (report) | - |
| nucleus-count Pearson r | 0.9978 | >= 0.90 | yes |
| per-nucleus intensity r, FOXA2-488 | 0.9987 | >= 0.90 | yes |
| per-nucleus intensity r, NANOG-594 | 0.9996 | >= 0.90 | yes |

Nuclei: 1218 (reference) vs 1233 (ours). Cellpose device: **mps (GPU)**. **RESULT: PASS.**
Timing: reference 4.6 s total (~0.4 s/img, CPU-TF); Cellpose-SAM 121 s (~11 s/img, MPS).

---

## Dataset 3: ACTH D40
Grouped single-channel TIFF (DAPI + 488 + 594), 8-bit, 960x1280. Conditions FN-CNEG
(secondary-only control), FN-D40, R1-D40. Fields (15): `20X-FN-CNEG-ACTH-Foto1..5`,
`20X-FN-D40-ACTH-Foto1..5`, `20X-R1-D40-ACTH-Foto1..5`.
Dice 0.961, F1@IoU.5 0.962, matched IoU 0.907, count r 0.9973 (3173 vs 3151 nuclei),
intensity r 488 0.9988 / 594 0.9983. Device mps. Reference 8.5 s / Cellpose 164 s. **PASS.**

## Dataset 4: IF FOXA2/NANOG D0/D7/D15
Grouped single-channel TIFF (DAPI + 488 + 594), 8-bit, 960x1280; days D0/D7/D15 (some very dense,
1000+ nuclei/field). Fields (21): `20X-E6-D15-N14-FOXA2-Foto1..5`, `20X-FN-D0-N14-FOXA2-Foto1..5`,
`20X-FN-D7-N14-FOXA2-Foto1..5`, `20X-FN-D15-N14-FOXA2-Foto1..6`.
Dice 0.948, F1@IoU.5 0.942, matched IoU 0.881, count r 0.9996 (10729 vs 10663 nuclei),
intensity r 488 0.997 / 594 0.9977. Device mps. Reference 15 s / Cellpose 230 s. **PASS.**

## Dataset 5: EpCam Dia 60 (from CZI)
15 CZI files (converted to per-channel TIFF via our loader; channels DAPI/AF488/RP647, 16-bit,
1920x1216). Conditions FND60-EpcamPE, FND60-control2rio (secondary-only), R1D60-EpcamPE. Fields
(15): `20X-FND60-EpcamPE-Foto1..5`, `20X-FND60-control2rio-Foto1..5`, `20X-R1D60-EpcamPE-Foto1..5`.
Dice 0.964, F1@IoU.5 0.949, matched IoU 0.921, count r 0.9914 (2403 vs 2304 nuclei),
intensity r AF488 0.9985 / RP647 0.9983. Device mps. Reference 12.8 s / Cellpose 472 s
(these are the largest images, 16-bit full-res, so Cellpose ~31 s/field). **PASS.**

## Conclusion
On all FIVE datasets CelloType's results agree closely with the reference nuclear analysis:
segmentations are highly concordant (Dice ~0.95, F1@IoU>=0.5 0.94-0.97, matched IoU ~0.88-0.92) and
per-nucleus marker readouts are highly correlated (r ~= 0.997-0.9996). The main practical difference is speed:
the reference StarDist model (small, here on CPU) is ~15-40x faster than Cellpose-SAM (a large
transformer on MPS), most pronounced on the large 16-bit images. Agreement is not accuracy: neither
method was compared with manual annotations, so this study cannot say which one is closer to the
true nuclei where they differ. Cellpose
ran on GPU/MPS in every run (confirmed by the harness printing/checking the device). EpCam CZI
decoded correctly (no all-zero files).

## Reproduce
The comparison harness segments the DAPI channel of each field with both methods, removes objects
within 3 px of the border from both, matches nuclei one-to-one, and measures every marker with the same code. It
writes per-image metrics (`per_image_metrics.csv`) and an aggregate summary (`summary.json`) with
PASS/FAIL against the agreement thresholds above. Reference method: StarDist (`2D_versatile_fluo`) via a
published ImageJ/FIJI multichannel nuclear-analysis plugin; ours: Cellpose-SAM (`cpsam`) on the GPU.
