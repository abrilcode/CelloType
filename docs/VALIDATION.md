# Agreement study: CelloType (Cellpose-SAM) vs a published StarDist workflow

Goal: check that CelloType's nuclear segmentation and per-nucleus marker measurements agree with
an established, published nuclei-analysis workflow, on the same images, automatically, with
quantitative metrics and timing. Tested on five datasets; all met the agreement thresholds set
before the comparison.

**Scope.** This is an agreement study between two automatic methods. The reference is another
automatic segmentation, not a manual annotation, so the results show that both methods give
concordant segmentations and readouts on these images; they do not establish the accuracy of
either method. Every Cellpose-SAM run used the GPU (Apple MPS) at full resolution (no
downscaling); the StarDist reference ran on the CPU. No run used background subtraction, and only
nuclei were segmented.

**StarDist option in the app.** The app can also segment nuclei with StarDist. That option runs
the same model and settings as the reference below (on field `20X-FND60-EpcamPE-Foto1` of
dataset 5 its labels are identical to the reference call, 181 objects). The agreement figures in
this document describe Cellpose-SAM against that reference; results made with the StarDist option
are those of the reference method.

## Method (shared by all datasets)

**Reference method, reproduced exactly** from the published Multi-Channel Nuclear Analysis macro
for Fiji (`Multi_Channel_Nuclear_Analysis.ijm`): StarDist model "Versatile (fluorescent nuclei)"
(`2D_versatile_fluo`), `normalizeInput` percentiles 1.0 to 99.8, `probThresh=0.5`,
`nmsThresh=0.4`, `excludeBoundary=2`, on the DAPI channel; no background subtraction. It was run
in Python with the `stardist` package in an isolated environment (there, `b=2` counts cells of the
model's 2 × 2 probability grid, about 4 px). The Fiji plugin wraps the same TensorFlow model, so
the segmentation is identical; running the Fiji interface headless is impractical and
unnecessary.

**CelloType's method**: `segment.segment_channel` (Cellpose-SAM, the `cpsam_v2` weights that are
the default model of the tested Cellpose 4.2.1.1) on the same DAPI channel, exactly as the app
runs it, on the GPU (Apple MPS); the harness prints and checks the device. On top of StarDist's
own boundary exclusion, objects with any pixel within 3 px of the image border (scikit-image
`clear_border`, `buffer_size=2`) are removed from both label sets. (The app itself now leaves out
objects touching the image border by default.) Both label sets are measured with the same code
(`measure.measure_field`), so any difference in the readout comes only from the masks.

**What is measured, and why:**
- Segmentation agreement: foreground **Dice** and **IoU** (do the nuclear regions overlap); object
  **count** per image (and Pearson r across images); instance **F1 at IoU ≥ 0.5** and **mean IoU
  of matched nuclei** (are they the same objects, one to one).
- Measurement agreement: **per-nucleus mean intensity** of each marker on matched nuclei → Pearson r.
- Timing: wall-clock time of the reference (StarDist) and of CelloType (Cellpose-SAM).

Agreement thresholds, set before the comparison: Dice ≥ 0.85, count r ≥ 0.90, F1 at IoU ≥ 0.5
≥ 0.70, per-marker intensity r ≥ 0.90.

## Summary (5 datasets, all within the thresholds)

| Dataset | Fields | Nuclei, reference / CelloType | Dice | F1 (IoU ≥ 0.5) | Matched IoU | Count r | Intensity r (markers) | Time, reference / CelloType (s) |
|---|---|---|---|---|---|---|---|---|
| Public example set | 16 | 3862 / 3768 | 0.950 | 0.949 | 0.897 | 0.9994 | 0.998 / 0.998 | 8.6 / 178 |
| Maru (FN/R1 D40) | 11 | 1218 / 1233 | 0.956 | 0.965 | 0.910 | 0.9978 | 0.999 / 0.9996 | 4.6 / 121 |
| ACTH D40 | 15 | 3173 / 3151 | 0.961 | 0.962 | 0.907 | 0.9973 | 0.999 / 0.998 | 8.5 / 164 |
| IF FOXA2/NANOG D0/D7/D15 | 21 | 10729 / 10663 | 0.948 | 0.942 | 0.881 | 0.9996 | 0.997 / 0.998 | 15 / 230 |
| EpCam Dia 60 (from CZI) | 15 | 2403 / 2304 | 0.964 | 0.949 | 0.921 | 0.9914 | 0.999 / 0.998 | 12.8 / 472 |

All five datasets meet every threshold: Dice about 0.95, F1 at IoU ≥ 0.5 from 0.94 to 0.97,
per-nucleus marker intensity r from 0.997 to 0.9996.

---

## Dataset 1: public example set
16 multichannel TIFF (Ch1 DAPI, Ch2 NANOG-488, Ch3 OCT6-mCherry), 8-bit, 960 × 1280, 4 conditions
× 4 fields. Images: `20X-Clon3-Indif-minusDox-D0-Foto1..4`, `20X-Clon3-Indif-plusDox-D1-Foto1..4`,
`...-plusDox-D2-Foto1..4`, `...-plusDox-D3-Foto1..4` (16 files).

| Metric | Value | Threshold | Within |
|---|---|---|---|
| Mean foreground Dice | 0.950 | ≥ 0.85 | yes |
| Mean instance F1 at IoU ≥ 0.5 | 0.949 | ≥ 0.70 | yes |
| Mean IoU of matched nuclei | 0.897 | (reported) | |
| Nucleus-count Pearson r | 0.9994 | ≥ 0.90 | yes |
| Per-nucleus intensity r, NANOG | 0.998 | ≥ 0.90 | yes |
| Per-nucleus intensity r, OCT6 | 0.998 | ≥ 0.90 | yes |

Nuclei: 3862 (reference) and 3768 (CelloType). Within all thresholds.
Timing: reference 8.6 s in total (about 0.5 s per image, CPU); Cellpose-SAM 178 s (about 11 s per
image, GPU).

---

## Dataset 2: Maru (FN/R1 D40)
Grouped single-channel TIFF per field (DAPI, FOXA2-488, NANOG-594), 8-bit, 960 × 1280; conditions
FN-D40 and R1-D40 (plus one FN-D15 DAPI-only field). Some fields are incomplete (FN-D15-Foto1 has
DAPI only; R1-D40-Foto1 has no FOXA2) and are handled without errors.
Fields (11): `20X-FN-D15-Foto1`, `20X-FN-D40-Foto1..5`, `20X-R1-D40-Foto1..5`.

| Metric | Value | Threshold | Within |
|---|---|---|---|
| Mean foreground Dice | 0.956 | ≥ 0.85 | yes |
| Mean instance F1 at IoU ≥ 0.5 | 0.965 | ≥ 0.70 | yes |
| Mean IoU of matched nuclei | 0.910 | (reported) | |
| Nucleus-count Pearson r | 0.9978 | ≥ 0.90 | yes |
| Per-nucleus intensity r, FOXA2-488 | 0.9987 | ≥ 0.90 | yes |
| Per-nucleus intensity r, NANOG-594 | 0.9996 | ≥ 0.90 | yes |

Nuclei: 1218 (reference) and 1233 (CelloType). Within all thresholds.
Timing: reference 4.6 s in total (about 0.4 s per image, CPU); Cellpose-SAM 121 s (about 11 s per
image, GPU).

---

## Dataset 3: ACTH D40
Grouped single-channel TIFF (DAPI, 488, 594), 8-bit, 960 × 1280. Conditions FN-CNEG
(secondary-only control), FN-D40, R1-D40. Fields (15): `20X-FN-CNEG-ACTH-Foto1..5`,
`20X-FN-D40-ACTH-Foto1..5`, `20X-R1-D40-ACTH-Foto1..5`.
Dice 0.961, F1 at IoU ≥ 0.5 0.962, matched IoU 0.907, count r 0.9973 (3173 and 3151 nuclei),
intensity r 488 0.9988 / 594 0.9983. Reference 8.5 s, Cellpose-SAM 164 s. Within all thresholds.

## Dataset 4: IF FOXA2/NANOG D0/D7/D15
Grouped single-channel TIFF (DAPI, 488, 594), 8-bit, 960 × 1280; days D0/D7/D15 (some very dense,
over 1000 nuclei per field). Fields (21): `20X-E6-D15-N14-FOXA2-Foto1..5`,
`20X-FN-D0-N14-FOXA2-Foto1..5`, `20X-FN-D7-N14-FOXA2-Foto1..5`, `20X-FN-D15-N14-FOXA2-Foto1..6`.
Dice 0.948, F1 at IoU ≥ 0.5 0.942, matched IoU 0.881, count r 0.9996 (10729 and 10663 nuclei),
intensity r 488 0.997 / 594 0.9977. Reference 15 s, Cellpose-SAM 230 s. Within all thresholds.

## Dataset 5: EpCam Dia 60 (from CZI)
15 CZI files read with the app's loader (channels DAPI, AF488, RP647; 16-bit, 1920 × 1216).
Conditions FND60-EpcamPE, FND60-control2rio (secondary-only), R1D60-EpcamPE. Fields (15):
`20X-FND60-EpcamPE-Foto1..5`, `20X-FND60-control2rio-Foto1..5`, `20X-R1D60-EpcamPE-Foto1..5`.
Dice 0.964, F1 at IoU ≥ 0.5 0.949, matched IoU 0.921, count r 0.9914 (2403 and 2304 nuclei),
intensity r AF488 0.9985 / RP647 0.9983. Reference 12.8 s, Cellpose-SAM 472 s (the largest
images, 16-bit at full resolution, about 31 s per field). Within all thresholds.

## Conclusion
On all five datasets CelloType's results agree closely with the reference nuclear analysis:
segmentations are highly concordant (Dice about 0.95, F1 at IoU ≥ 0.5 from 0.94 to 0.97, matched
IoU from 0.88 to 0.92) and per-nucleus marker readouts are highly correlated (r from 0.997 to
0.9996). The main practical difference is speed: the reference StarDist model (small, run on the
CPU) is 15 to 37 times faster than Cellpose-SAM (a large transformer, run on the GPU), most
markedly on the large 16-bit images. Agreement is not accuracy: neither method was compared with
manual annotations, so this study cannot say which one is closer to the true nuclei where they
differ. The EpCam CZI files were decoded correctly (no all-zero channels).

## Comparison harness
The harness segments the DAPI channel of each field with both methods, removes objects within 3 px
of the border from both, matches nuclei one to one, and measures every marker with the same code.
It writes per-image metrics and an aggregate summary checked against the agreement thresholds
above. The harness and its per-image results are not part of this repository.
