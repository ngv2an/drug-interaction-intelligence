# Day 2 Notes - Reproducible Smoke Run

## Muc tieu ngay 2

Ngay 2 tap trung vao tinh tai lap cua mot lan chay nho:

- CLI ro rang hon.
- Moi lan chay co `run_id`.
- Luu config va metrics ra file JSON.
- Output sinh ra nam trong `artifacts/` va khong dua vao Git.

Day la buoc MLOps dau tien: khong chi "model chay duoc", ma phai biet lan chay do dung input nao, seed nao, version moi truong nao, va metric nao.

## Lenh chay

```bash
my_env/bin/python 'src/DDI_prediction_experiment 20160716.py' --smoke-test --sample-size 50 --seed 0 --run-name day2_smoke_seed0_n50
```

Neu khong truyen `--run-name`, code se tu tao ten folder theo seed, sample size va timestamp UTC.

## Output

Output duoc luu tai:

```text
artifacts/day2_runs/day2_smoke_seed0_n50/
```

Trong do co:

```text
run_config.json
metrics.json
```

`run_config.json` ghi lai:

- `run_id`
- `run_type`
- `seed`
- `sample_size`
- `test_pairs`
- dataset path
- model dang dung
- python/numpy/networkx/scikit-learn/deap version

`metrics.json` ghi lai:

- AUC
- AUPR
- Precision
- Recall
- Accuracy
- F1

## Ket qua kiem tra ngay 2

```text
AUC: 0.8966
AUPR: 0.6260
Precision: 0.5517
Recall: 0.6076
Accuracy: 0.9229
F1: 0.5783
Test pairs: 908
```

## Ghi chu thuc te

- `artifacts/` da duoc them vao `.gitignore` vi day la output sinh ra khi chay experiment.
- Git chi nen luu source code, config mau, docs va tests. Run artifacts nen luu bang artifact store, MLflow, S3, MinIO hoac thu muc local bi ignore.
- Ngay 3 chung ta se bat dau tach cac ham nho de test duoc, nhung van dua tren file legacy nay.
