# Day 1 Notes - Legacy DDI Baseline

## Muc tieu ngay 1

Ngay 1 chi tap trung lam cho code goc co mot baseline nho chay duoc. Chua refactor sau, chua them API, chua Docker.

Code goc duoc dung lam source of truth:

```text
src/DDI_prediction_experiment 20160716.py
```

File `src/ddi_prediction.py` khong duoc dung lam nen cho huong phat trien nay.

## Flow cua code goc

1. `load_csv()` doc cac ma tran trong `dataset/`.
2. `cross_validation()` chia cac canh DDI da biet thanh train/test theo fold.
3. `internal_determine_parameter()` tao them holdout noi bo de hoc trong so ensemble.
4. `ensemble_method()` sinh nhieu ma tran du doan tu:
   - similarity feature: chemical, target, transporter, enzyme, pathway, indication, side effect, off-side effect
   - topology feature: common neighbors, Adamic-Adar, resource allocation, Katz, ACT, RWR
   - label propagation
   - disturb matrix
5. `getParamter()` dung genetic algorithm cua DEAP de hoc trong so ensemble.
6. `ensemble_scoring()` ket hop cac base model va tinh metric.
7. Ket qua cu duoc ghi ra thu muc `result/`.

## Nhung diem da sua de code chay duoc voi moi truong hien tai

- Comment import cu `from pylab import *`, thay bang NumPy truc tiep de khong can cai them matplotlib.
- Comment `nx.from_numpy_matrix`, thay bang `nx.from_numpy_array` vi NetworkX ban moi da doi API.
- Comment cach dung `MinMaxScaler.fit_transform()` tren vector 1D, thay bang helper reshape ve 2D roi flatten lai.
- Them guard cho `deap.creator.create()` de khong loi khi goi GA nhieu lan trong cung process.
- Them solver `liblinear` cho logistic regression L1 vi scikit-learn ban moi yeu cau solver phu hop.
- Reshape input cua `predict_proba()` ve shape `(1, n_features)`.
- Boc doan full run cu bang `main()` de import file khong tu dong chay experiment 20 lan.

## Lenh smoke test

Chay baseline nho:

```bash
my_env/bin/python 'src/DDI_prediction_experiment 20160716.py' --smoke-test --sample-size 50 --seed 0
```

Ket qua ngay 1 tren sample 50:

```text
AUC: 0.8966
AUPR: 0.6260
Precision: 0.5517
Recall: 0.6076
Accuracy: 0.9229
F1: 0.5783
Test pairs: 908
```

## Full experiment legacy

Full run van duoc giu lai, nhung khong nen chay trong ngay 1 vi mat thoi gian:

```bash
my_env/bin/python 'src/DDI_prediction_experiment 20160716.py' --legacy-full-run --runtimes 20 --cv-num 5
```

Trong cac ngay sau, chung ta se tach dan code thanh CLI/experiment tracking/API, nhung moi buoc se dua tren file legacy nay.
