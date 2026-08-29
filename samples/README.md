# Sample images

Source: flower_image_33.jpg -> It is already present in the backend/data/clean and is a subset sample taken from the dataset "https://www.kaggle.com/datasets/prasunroy/natural-images". Therefore it is a REAL IMAGE.

Seven variants of the same source image, each showing one quality condition
in isolation, generated via `app/ml/degrade.py`'s degradation functions at a
fixed severity so they're directly comparable side by side.

| File | Condition |
|---|---|
| `sample_1_clean.jpg` | Clean / acceptable baseline |
| `sample_2_blur.jpg` | Blur (Gaussian, severity 0.75) |
| `sample_3_underexposed.jpg` | Underexposure (severity 0.7) |
| `sample_4_overexposed.jpg` | Overexposure (severity 0.6) |
| `sample_5_noise.jpg` | Sensor noise (Gaussian + speckle, severity 0.6) |
| `sample_6_corruption.jpg` | Corruption (block dropout / re-encoding artifacts, severity 0.7) |
| `sample_7_defect.jpg` | Localized visual defect (scratch/spot/occlusion, severity 0.6) |

