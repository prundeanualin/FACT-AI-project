# FACT-AI-project
The project for the FACT-AI MSc course at UvA. Follow instructions below to run the FACT-AI-project FairLISA code:

---

## Datasets

**MovieLens1M:**

Download the zip file: https://grouplens.org/datasets/movielens/1m/

Make sure your MovieLens1M data is in:

- data/ml-1m/movies.dat
- data/ml-1m/ratings.dat
- data/ml-1m/users.dat

**PISA2015:**

Under "SAS (TM) Data Files (compressed)" or under "SPSS (TM) Data Files (compressed)": https://www.oecd.org/pisa/data/2015database/

Download both the "Student questionnaire data file" and the "Cognitive item data file".

Make sure your PISA2015 data is in:

- data/pisa2015/PUF_SAS_COMBINED_CMB_STU_COG/cy6_ms_cmb_stu_cog.sas7bdat
- data/pisa2015/PUF_SAS_COMBINED_CMB_STU_QQQ/cy6_ms_cmb_stu_qqq.sas7bdat
- data/pisa2015/PUF_SAS_COMBINED_CMB_STU_QQQ/cy6_ms_cmb_stu_qq2.sas7bdat

Or:

- data/pisa2015/PUF_SPSS_COMBINED_CMB_STU_COG/CY6_MS_CMB_STU_COG.sav
- data/pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ/CY6_MS_CMB_STU_QQQ.sav
- data/pisa2015/PUF_SPSS_COMBINED_CMB_STU_QQQ/CY6_MS_CMB_STU_QQ2.sav

**Run if you want to create the directories for ease:**
```
python setup_directories.py
```
**Preprocessing datasets:**
```
python preprocess_dataset_movielens.py
```
---
# Recommender Systems (MovieLens1M)

## User models

By default, use the explicit feedback scenario:

### Train PMF (explicit)
```
python pmf_explicit.py
```
### Train PMF (implicit)

pmf_explicit.py will train the model, pmf_embeddings.py will save the embeddings to .csv which is only needed for implicit feedback due to our setup

**Binarize threshold 1:**
```
python pmf_explicit.py -IMPLICIT True

python pmf_embeddings.py
```
**Binarize threshold 3:**
```
python pmf_explicit.py -IMPLICIT True -BINARIZATION_THRESHOLD 3

python pmf_embeddings.py -BINARIZATION_THRESHOLD 3
```
### Train NCF (explicit)
```
python ncf_explicit.py
```
### Train NCF (implicit)

**Binarize threshold 1:**
```
python ncf.py

python ncf_embeddings.py
```
**Binarize threshold 3:**
```
python ncf.py -BINARIZATION_THRESHOLD 3

python ncf_embeddings.py -BINARIZATION_THRESHOLD 3
```
---

## FairLISA (Single Run) - Recommender Systems (MovieLens1M)

Change RESULTS_FILENAME accordingly. Results will be appended to this file.

**Default parameters:**

(Explicit only) For NCF_MODEL_PATH we use ncf_models_explicit/ncf_model_explicit_emb_16_epoch_30.pth because it performed best on the validation set

(Explicit only) For PMF_MODEL_PATH we use pmf_models/pmf_model_explicit_emb_16_epoch_80.pth because it performed best on the validation set

MISSING_RATIO is set to 0.2 by default.

You can specify the user model MODEL as either PMF or NCF.

Lambdas are LAMBDA_1 1 LAMBDA_2 20 LAMBDA_3 10 by default for FairLISA.

You can change ORIGIN to True to skip Filter training and get Origin results.

You can set LAMBDA_3 to 0 for ComFair.

You can change the SEED.

(Implicit only) You should set BINARIZE_THRESHOLD to 1 or 3 or any other (between 1 and 5) depending what you used to train the implicit user models

**Run:**
```
python LISA_explicit.py -USE_TEST_DATA True -SAVE_RES_TO_TXT True -RESULTS_FILENAME "ml_explicit_single_run_results"

python LISA_implicit.py -BINARIZE_THRESHOLD 1 -USE_TEST_DATA True -SAVE_RES_TO_TXT True -RESULTS_FILENAME "ml_implicit_single_run_results"
```
---

## FairLISA Batched Experiments - Recommender Systems (MovieLens1M)

Results are placed in ml_experiments_results.

Missing ratios experiments will run for PMF and NCF all mising ratios and apply all three fairness frameworks ComFair, FairLISA, Origin.

Lambdas experiments will run for PMF and NCF mising ratio 0.4 and apply all three fairness frameworks ComFair, FairLISA, Origin while fixing one labmda (at 0 or 15) and varying the other (0, 5, 10, 15)

All experiments marked with MAIN reproduce the FairLISA paper's results. EXTRA means the experiments go beyond the original paper. The tables and plots in our main text come from MAIN experiments.

### Explicit Missing ratios
Make sure your user models PMF and NCF explicit are trained.

#### Original experiment (MAIN):
```
python run_explicit_ml_missing_ratios.py
```
Run this for processing results .txt into plots/tables:
```
python process_ml_explicit_missing_ratios_results.py
```
#### Overfitted experiment (EXTRA):
```
python run_explicit_ml_missing_ratios.py -OVERFITTED True
```
Run this for processing results .txt into plots/tables:
```
python process_ml_explicit_missing_ratios_results.py -OVERFITTED True
```
### Implicit Missing ratios (EXTRA)
Make sure your user models PMF and NCF implicit are trained, with BINARIZATION_THRESHOLD 1 and 3.

#### Binarize threshold 1 experiment (EXTRA):
```
python run_implicit_ml_missing_ratios.py
```
Run this for processing results .txt into plots/tables:
```
python process_ml_implicit_missing_ratios_results.py
```
#### Binarize threshold 3 experiment (EXTRA):
```
python run_implicit_ml_missing_ratios.py -BINARIZATION_THRESHOLD 3
```
Run this for processing results .txt into plots/tables:
```
python process_ml_implicit_missing_ratios_results.py -BINARIZATION_THRESHOLD 3
```
### Explicit lambdas (MAIN)
```
python run_explicit_ml_lambdas.py
```
Run this for processing results .txt into plots/tables:
```
python process_ml_explicit_lambdas.py
```
---
# Cognitive Diagnosis (PISA2015)

For running the cognitive diagnosis experiments, the following parameters can be tweaked:


```
python <experiment_file>.py -MODEL MIRT -TRAIN_USER_MODEL True -PREPROCESS_DATA True -SEED 420
```

- The default user model is MIRT. You can change the underlying user model to one of [IRT, MIRT, NCDM]
- You can set to train the user model from the scratch, if it has not been trained before
- You can set to perform data preprocessing, if it has not been done before
- You can change the seed value (default is 420)

## Run the missing ratios experiments

```
python run_experiment_missing_ratios.py
```

All 3 fairness frameworks (Origin, ComFair and FairLISA) are tested with different ratios of data without sensitive labels.
The missing raios are [0.2, 0.4, 0.6, 0.8, 0.95]. The corresponding plots are also generated afterwards.

---

## Run the ablations for lambda2 and lambda3

```
python run_experiment_lambdas.py
```
For the FairLISA fairness framework, the fairness performance is tested with both lambda values (lambda1, lambda2)
varying between [0, 0.5, 1, 1.5] while the other is kept fixed at 0. The corresponding plots are also generated afterwards.

---

# From the authors' README:

## FairLISA
Code for the NeurIPS'2023 paper "FairLISA: Fair User Modeling with Limited Sensitive Attributes Information"

## Run
```
python3 run.py -DATA your_dataset -FILTER_MODE separate -FAIRNESS_RATIO 1.0 -FAIRNESS_RATIO_NOFEATURE 0.5 -CUDA 5 -USER_NUM 358415 -ITEM_NUM 183 -KNOWLEDGE_NUM 16 -LATENT_NUM 16 -MODEL IRT -NO_FEATURE 0.6 -USE_NOFEATURE True
```

### BibTex
If you find this work useful in your research, please cite our paper:

```
@inproceedings{zhang2023fairlisa,
  title={FairLISA: Fair User Modeling with Limited Sensitive Attributes Information},
  author={Zhang, Zheng and Liu, Qi and Jiang, Hao and Wang, Fei and Zhuang, Yan and Wu, Le and Gao, Weibo and Chen, Enhong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
