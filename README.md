# Probabilistic ratio cut

## Environment setup

-   Create a python environment and make sure that poetry package is installed
-   Activate the python environment and from within the code folder call
    `poetry install`. This should install all the dependencies and the package
-   For producing the Turtle results, we use the author's implementation, which
    should be cloned to the same folder for `run_turtle.py` to work

```bash
git clone https://github.com/mlbio-epfl/turtle
## This step is important to be able to import turtle as a package
touch turtle/__init__.py
```

## Computing representations

There are 2 representation types we can extract:

-   the ones that come from Dino and CLIP:

```bash
python extract_vit_representation.py --dataset cifar10 --model dinov2 --root-dir /buckets/ml --device-num 1
```

This will create the folder `/buckets/ml/representations/dinov2/` that contains
`cifar10_[feats/y]_[train/val].npy`

-   the ones that come from solo Learn package

```bash
python extract_solo_representation.py method=dino --config-name solo_cifar100
```

This will behave similarly to the previous call. The available representations
are in `prcut.data.solo_checkpoints`

## Training the models

Once the different representations are extracted, we can call the following to
run PRCut:

```bash
python run_turtle.py --dataset fashionmnist --phis dinov2 --root_dir /buckets/ml/
python run_prcut.py dataset=fashionmnist repr=dinov2 root_dir=/buckets/ml/

```

or Turtle:

```bash
python run_turtle.py --dataset fashionmnist --phis dinov2 --root_dir /buckets/ml/

```
