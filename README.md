# RASC-FCM reproducibility

Code and complete experimental records for **Evaluating Rare-Class-Aware Consensus Feature Selection for Federated IoT Intrusion Detection**.

## Evidence included

- 420 neural test-run records: 320 optimizer, heterogeneity, external-cohort and duration runs, plus 100 fixed-budget selector-control runs.
- 36 validation-only optimizer-tuning records and the 20-seed controlled rare-signal fixture.
- 420 selected parameter checkpoints, exact probability hashes, deterministic seeds, masks, partitions and row-selection manifests.
- Scripts for acquisition, preprocessing, retraining, checkpoint replay, all statistical comparisons and publication figures.

The complete scientific archive is deposited at https://doi.org/10.5281/zenodo.23765655. The source repository is https://github.com/classworkneet-rgb/RASC-FCM-Reproducibility. Download and extract the Zenodo ZIP for the complete records, checkpoints, row-ID selections and checksum manifest. GitHub contains the essential implementation, configurations and data-acquisition manifests; the full experimental records are in the DOI archive.

## Environment and integrity

Use Python 3.12 with the pinned dependencies in `requirements.txt`. Before generating outputs:

    python -m pip install -r requirements.txt
    python src/verify_release.py
    python -m unittest discover -s tests -v

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for deterministic execution. Environments are recorded in `configs/execution_environment.json` and `configs/selector_execution_environment.json`.

## Recompute tables and statistics without traffic data

    python src/summarize.py
    python src/additional_inference.py
    python src/analyze_selector_extension.py
    python src/make_figures.py
    python src/make_selector_figure.py
    python src/run_controlled.py

`results/manuscript_tables/` contains data for Tables 1–10. `results/selector_extension_contrasts.csv` contains every selector contrast, pointwise interval and Holm-adjusted p-value. The original inference families are unchanged. The ten selector contrasts form a separate exploratory family specified after the reference outcomes were known. The 200-round follow-up also remains exploratory.

## Acquire provider data and replay checkpoints

Traffic-feature vectors and raw traffic archives are not redistributed. Read `docs/DATA_TERMS.md` and acquire the public provider cohorts with frozen hashes:

    python src/acquire_data.py --dataset bot
    python src/prepare_bot.py
    python src/acquire_data.py --dataset cic
    python src/prepare_cic.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/replay_checkpoints.py --restore-run-arrays
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/verify_results.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/verify_final_contracts.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/verify_selector_extension.py

The first replay verifies the 320 reference runs. The selector verifier reconstructs all 100 additional predictions, checks their exact float32 hashes and recomputes every metric. Full probability matrices are reproducible from the checkpoints and are not stored redundantly. The optional `--retrain-interface` argument also repeats the two reference training trajectories used to validate the selector interface.

## Resume or retrain

After cohort reconstruction:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/run_release.py --phase all --workers 4
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/selector_extension.py --freeze-masks
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/selector_extension.py --workers 4

The selector-freezing command recomputes all rankings from training rows and verifies every complete RASC mask against the reference records. Existing frozen masks are checked rather than replaced. The five alternatives are omission of stability, uniform supported-class weights, omission of reserved slots, client-aggregated mutual information and seeded random selection, each at 22/37 features under FedAvg and FedAdam.

Those commands resume completed records. For a complete new training run after acquiring the provider cohorts, first create a separate execution tree:

    python src/create_retraining_copy.py --destination ../RASC_FCM_Retrain

Change to that new directory and run the three training commands above, followed by the analysis and replay commands. The helper refuses an existing destination, removes cached result records only from the new copy, and retains the frozen reference checkpoints for independent comparison. Training does not load those checkpoints. Verify the original package checksum before generating new outputs; runtime fields in a fresh run are expected to differ.

## Scientific scope and reuse

The implementation is an offline simulation with pooled training-only preprocessing. It does not establish physical-device scalability, differential privacy or cryptographic secure aggregation. The BoT-IoT split is chronological within category, with retained ports and residual dependencies. CICIoT2023 uses a bounded public-mirror cohort and does not establish cross-dataset transfer or capture-disjoint generalization. Paired seeds do not constitute independent captures.

Source code is MIT licensed; author-produced records and checkpoints are CC BY 4.0. Dataset rights remain with the providers. See `LICENSE_CODE.txt`, `docs/REUSE_LICENSES.md` and `CITATION.cff`.
