# Day 3 Notes - Testable Core

## Muc tieu ngay 3

Ngay 3 chi lam "testable core": tach mot so logic nho ra package moi va viet unit tests. Chua them config CLI, chua MLflow, chua API.

Legacy file van la source of truth:

```text
src/DDI_prediction_experiment 20160716.py
```

Package moi:

```text
src/drug_interaction_intelligence/
```

Package nay dung de phat trien sach dan ma khong pha file legacy.

## Nhung phan da tach

```text
paths.py           # resolve path tu project root
data.py            # load matrix CSV, validate square matrix, load drug ids
preprocessing.py   # ep vector ve 1D float, min-max normalize
split.py           # collect link/non-link, holdout split theo link
metrics.py         # convert metric list sang dict, collect score theo test positions
```

## Vi sao can testable core?

Code research cu thuong viet theo kieu script dai, kho test tung phan. MLOps can cac ham nho co the test doc lap de:

- Giam loi khi refactor.
- Biet bug nam o data, split, metric hay model.
- Chay CI sau nay.
- Giai thich code tot hon khi phong van.

## Tests

Da them `requirements-dev.txt` voi `pytest`.

Chay test:

```bash
my_env/bin/python -m pytest
```

Hien tai co 10 tests:

- Load matrix CSV dung shape.
- Validate square matrix.
- Load drug ids.
- Normalize vector ve `[0, 1]`.
- Flatten vector ve 1D float.
- Collect link/non-link tren upper triangle.
- Holdout split xoa canh test doi xung.
- Convert metrics list sang dict.
- Collect labels/scores theo test positions.

Ket qua:

```text
10 passed
```

## Baseline check

Legacy smoke run van pass:

```bash
my_env/bin/python 'src/DDI_prediction_experiment 20160716.py' --smoke-test --sample-size 50 --seed 0 --run-name day3_legacy_smoke_check
```

Metric van giu:

```text
AUC: 0.8966
AUPR: 0.6260
Precision: 0.5517
Recall: 0.6076
Accuracy: 0.9229
F1: 0.5783
```

## Gioi han ngay 3

Chua them config CLI. Do la viec cua Day 4.
