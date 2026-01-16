# AttMAMLToxIL
 
Few-shot Prediction of Toxicity of Ionic Liquids Supported by Att-MAML: Attentive Model-Agnostic Meta-Learning

Corresponding author: Karol Baran (GdańskTech), karol.baran[at]pg.edu.pl

Manuscript status: submitted (2026)

## Files and folders:

- data - directory with information on data and scripts to scrap data
- codes - directory with codes used in the study

## Instructions: how to run the codes?

1. Please change directory to data (`cd data`) and create `Data.xlsx` file in data directory as descriped in data/README.md. 
1. Change directory into codes (e.g. `cd ../codes`). 
1. Install dependencies using your preferred tool like conda or uv. If using conda, one could use: `conda env create --file=envdep.yaml` and `conda activate metadeep`.
1. Revisit config_dict variable in run_experiments.py. 
1. Run `python3 run_experiments.py`

## Authors:

Karol Baran, Adam Kloskowski (GdańskTech)

2026 Gdańsk, Poland

repository maintained by Karol Baran

## Acknowledgment:

The authors would like to gratefully acknowledge that this research was funded in whole or in part by the National Science Centre, Poland (NCN) under the Preludium 22 program in the years 2024–2027 (project no. UMO-2023/49/N/ST5/01043).
