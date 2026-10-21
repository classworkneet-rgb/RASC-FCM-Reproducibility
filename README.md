# RASC FCM reproducibility code

Text source code, frozen manifests, seeds, masks, selected settings and checksums for rare-class-aware consensus feature selection in federated IoT intrusion detection. Detailed experimental records and binary artifacts are supplied in the complete research package separately.

## Frozen protocol and manifests

- `configs/protocol_freeze.json` records the protocol, seed sets, tuning grids and follow-up timing.
- `configs/selected_optimizer_settings.json` supplies selected settings without validation scores.
- `configs/seed_mask_manifest.json` supplies run configurations, ordered masks and partition digests, with no performance records.
- `data/*_manifest.json` and `data/frozen_artifact_checksums.json` supply raw-data, processed-array and partition checksums. Actual source-row, split, client and feature arrays are not public in this snapshot.
- `SHA256SUMS.txt` verifies this source snapshot.

The code does not implement differential privacy or cryptographic secure aggregation. Simulated clients do not establish physical device independence. Dataset terms and attribution remain in `docs/DATA_TERMS.md`; the code license does not relicense data.

## Reproduce

Use Python 3.12 and the pinned scientific packages.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/verify_release.py
python src/acquire_data.py --dataset bot
python src/prepare_bot.py
python src/acquire_data.py --dataset cic
python src/prepare_cic.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/run_release.py --phase all --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/verify_results.py
python src/summarize.py
python src/make_figures.py
```

Acquisition rejects changed provider bytes. Input arrays are reconstructed from the frozen source and preprocessing code. The original recorded masks and partition digests enable comparison with regenerated inputs. Result and figure scripts require completion of input preparation and training; detailed result JSON, models and probability arrays are excluded from this export. The complete package permits direct replay without raw downloads. Historical replay checks additionally require its legacy evidence file.

Scientific interpretation and residual split/feature dependencies are documented in the manuscript and complete archive. A new split, mask rule or optimization choice must be recorded as a new experiment. The historical source class name `FixedGatedDBNCNN` denotes a classifier that pretrains only its first dense layer with an RBM.

Repository: https://github.com/classworkneet-rgb/RASC-FCM-Reproducibility
