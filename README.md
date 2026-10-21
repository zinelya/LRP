# LRP: Lesion Radial Profiler

> **Early development.** The code is not yet stable.
> First release planned for december 2026. The goal is for it to be a ready to use flow from bids structure.

**Confluence-aware profiling of paramagnetic rim lesions (PRLs).**
LRP separates confluent PRLs and profiles quantitative MRI metrics across the core, rim and
peri-lesional compartments of each lesion.

Presented at the 10th Joint ACTRIMS-ECTRIMS Meeting (MSToronto 2026).

## Why

Paramagnetic rim lesions are an in vivo marker of chronic active lesions in multiple sclerosis.
PRLs are often confluent with neighbouring lesions, which makes compartment-level analysis
(core vs rim) difficult at scale. LRP is designed to separate confluent PRLs so each lesion can
be analysed individually, with any quantitative MRI metric.

## What it does

1. Segments lesions on FLAIR and T1w (LST-AI) and groups them into connected components.
2. An expert rater identifies the clusters that contain PRLs.
3. Confluent PRLs are separated by a watershed combining QSM, FLAIR and T1w.
4. PRL status is validated and the paramagnetic rim is segmented on QSM (Laplacian filter).
5. Compartments are defined and metrics are extracted in each:

## Acknowledgements

Sherbrooke Connectivity Imaging Laboratory (SCIL), Université de Sherbrooke; built on
[SCILPY](https://github.com/scilus/scilpy).
