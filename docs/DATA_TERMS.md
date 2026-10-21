# Data provenance and terms

BoT-IoT authoritative source: https://research.unsw.edu.au/projects/bot-iot-dataset . The provider grants free use for academic research and requires separate agreement for commercial use. Cite Koroniotis et al., Future Generation Computer Systems 100 (2019) 779–796, doi:10.1016/j.future.2019.05.041. Raw CSVs were acquired through the public mirror named in the frozen manifest. Four content hashes match the supplied Stage 6 evidence. No institutional digital signature is asserted.

CICIoT2023 authoritative source: https://www.unb.ca/cic/datasets/iotdataset-2023.html . Cite Neto et al., Sensors 23 (2023) 5941, doi:10.3390/s23135941. The actual acquired files are the train/validation/test derivative distributed by Kaggle user himadri07, with the published 46-feature schema. Their sizes and SHA-256 hashes are recorded. They comprise 7,845,673 rows, not the complete institutional dataset. Mirror redistribution does not establish byte identity to that complete archive or transfer the original provider's rights.

The complete research release includes processed experimental feature cohorts for research replay, labels, row IDs and the fitted transformations. These derived data remain subject to applicable provider terms and attribution. They are not relicensed under the code license. Raw archives are excluded. The public-repository export excludes raw and processed feature rows; it includes immutable row selections, acquisition/preparation code and content checksums.

Edge-IIoTset was acquired for a failed feasibility check, not used for a claimed external classification result and not redistributed here. The alternative CICIoT2023 path was selected before external training.

Global BoT-IoT row IDs concatenate CSV files 1 through 4 in file order. Global CIC row IDs concatenate the acquired train, validation and test files in that order. Per-split arrays are ordered by frozen class ID and then source ID (and category-time ordering in the supplied BoT transformation). Labels and IDs are stored alongside the arrays. Neither a random client partition nor a source row ID is a physical device/capture identifier.
