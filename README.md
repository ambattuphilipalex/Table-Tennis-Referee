# HouseCorr3D

## Setup

Clone the repository with all submodules:

```bash
git clone --recurse-submodules git@github.com:GenIntel/housecorr3d.git
cd housecorr3d
```

If you already cloned without submodules, initialise them:

```bash
git submodule update --init --recursive
```

Install all packages in editable mode:

```bash
bash setup/setup_slurm.sh
```

## Datasets

### Fetch (or place manually in your dataset path)

```bash
o3b dataset fetch -d hc3d_object
o3b dataset fetch -d dm_object
```

### Index

```bash
o3b dataset index -d hc3d_object
o3b dataset index -d hc3d_object_pair

o3b dataset index -d opentt
```

### Visualize

```bash
o3b dataset viz -d hc3d_object
o3b dataset viz -d hc3d_object_pair
o3b dataset viz -d hc3d_object --render

o3b dataset viz -d opentt
o3b dataset viz -d opentt --frames-per-scene 4
```

## Benchmark

### Run locally

```bash
o3b bench run -b hc3d_crsp3d_object_pair_nn -a category/housecorr3d_5
```

### Run on Slurm

```bash
o3b bench rrun -p slurm_lmbl40 -b hc3d_crsp3d_object_pair_nn -a category/housecorr3d_5
```
