# Five-dimensional kissing-number search

An exact 37-point kissing configuration extends the fourteen-contact anchor from Nyam-Osor's Table 1. The continuous 41-point search remains above the required inner-product threshold of (1/2). The unrestricted lower bound is 40 and Tomar's preprint gives an exact upper-bound certificate of 43. [Paper](paper.pdf) · [Anchor source](https://zenodo.org/records/24014108) · [Tomar's upper bound](https://www.mefyi.com/papers/kissing-numbers-five-seven/)

Run the witness and finite search with Python 3.10 or newer. The exact checker uses the standard library. Its mixed constraints compare rationals to square roots using integer squares with explicit sign checks. A separate reviewer replayed the 37-point witness using integer square-root enclosures.

```sh
python verify.py evidence/fixed-37.json
python lattice.py 1 grid-1.json
python lattice.py 2 grid-2.json --target 22
```

The radius-one exhaustive maximum is 11 new rays. At radius two, none of the 806 compatible integer rays can form a 22-clique. This excludes 37 total points within that finite radius-two search. The continuous 37-point witness uses integer directions outside the cube.

The numerical runs used Python 3.12.14 and the versions in `requirements.txt`. Use one BLAS thread to reproduce the search setup. Floating-point trajectories can vary across builds. The exact witness and integer lattice searches have no numerical dependencies.

```sh
python -m pip install -r requirements.txt
python search.py 37 fixed-37.json --southern --seed 6970 --restarts 96
python search.py 41 fixed-41.json --southern --seed 6970 --restarts 96
python flexible.py evidence/fixed-41.json flexible-41.json --seed 2308 --restarts 40
```

The best archived 41-point candidate from the runs with movable contacts has numerical maximum dot product about (0.51867). It fails the exact check. The published unrestricted 41-point search has reached a smaller value, also above (1/2). [Unrestricted comparison](https://aleckriebel.github.io/Math/research/kissing-number-5/)
