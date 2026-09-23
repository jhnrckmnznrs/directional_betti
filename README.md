# directional-betti

`directional-betti` computes exact sampled Betti tensors for three-parameter
cubical filtrations on 3-D voxel data. It also provides a frame-invariant
directional filtration for binary volumes.

The package has two public entry points:

```python
from directional_betti import (
    cubical_betti_tensors,
    frame_invariant_directional_betti,
)
```

The implementation is written in Python/NumPy with Numba kernels for the
connectivity sweeps.

## Installation

```bash
pip install directional_betti-1.0.0-py3-none-any.whl
```

Core dependencies are NumPy and Numba. TIFF input for the command-line
interface is optional:

```bash
pip install "directional-betti[io]"
```

## 1. Exact cubical Betti tensors from an arbitrary filtration

Let each top-dimensional voxel cube have a three-parameter filtration value

$$
F(x)=(F_1(x),F_2(x),F_3(x)).
$$

For a threshold vector $t=(t_1,t_2,t_3)$, a top cube is present when
$F_j(x)\le t_j$ for all three coordinates. Every lower-dimensional cubical
cell receives the coordinatewise minimum of the filtration values of its
incident top cubes. Thus edges, faces, and vertices can enter before any one
incident top cube enters.

For every sampled threshold triple the package computes

$$
B_q(t)=\dim H_q(K_t;\mathbb F_2),\qquad q=0,1,2,
$$

and the Euler characteristic $\chi(t)$.

A minimal example is:

```python
import numpy as np
from directional_betti import cubical_betti_tensors

X, Y, Z = 8, 7, 6
F = np.empty((X, Y, Z, 3), dtype=np.float32)
F[..., 0] = np.linspace(0.0, 1.0, X)[:, None, None]
F[..., 1] = np.linspace(0.0, 1.0, Y)[None, :, None]
F[..., 2] = np.linspace(0.0, 1.0, Z)[None, None, :]

result = cubical_betti_tensors(F, grid_steps=(8, 8, 8))

print(result.b0.shape)              # (9, 9, 9)
print(result.b0[-1, -1, -1])
print(result.b1[-1, -1, -1])
print(result.b2[-1, -1, -1])
```

An optional Boolean or nonzero-valued ROI can be supplied with shape
`(X, Y, Z)`. Top cubes outside the ROI are absent from the cubical complex.

The sampled grid uses `steps[j] + 1` thresholds on axis `j`. For each axis,
the minimum and maximum are taken over the valid top-cube filtration values and
the interval is divided uniformly.

### Exact topology computation

The implementation uses standard cubical-complex identities:

- $B_0$ is the number of connected components of the present cubical
  1-skeleton, computed by union-find on vertices and edges.
- $B_2$ is the number of bounded connected components of the complement,
  computed by a dual cubical graph with one distinguished exterior component.
- $B_1$ is obtained exactly from

$$
B_1=B_0+B_2-\chi.
$$

The returned tensors are checked for nonnegative Betti numbers and the exact
Euler--Poincare identity

$$
\chi=B_0-B_1+B_2.
$$

## 2. Frame-invariant directional filtration for binary volumes

For a binary foreground $B$ inside an ROI and a selected unit coordinate
direction $d\in\{e_x,e_y,e_z\}$, the package constructs

$$
F_d(x)=(C_d(x),A_d(x),S(x)).
$$

The three coordinates describe different local geometric properties.

### Cone occupancy deficit

Let $\Gamma_d(x)$ be a symmetric double cone centered at the voxel $x$,
with axial half-extent $H$ and radius $R_c$. The occupancy coordinate is

$$
C_d(x)
=
1-
\frac{|\Gamma_d(x)\cap B|}{|\Gamma_d(x)\cap \mathrm{ROI}|}.
$$

Thus small $C_d$ means that the directional cone is strongly occupied by
foreground voxels. The cone is clipped at the ROI boundary.

### Local covariance

Nearby foreground voxel centers are expressed relative to $x$ and assigned
nonnegative weights $w_x(y)$. The weighted centroid and covariance are

$$
\mu_x=
\frac{1}{W_x}\sum_y w_x(y)r_{xy},
\qquad
W_x=\sum_y w_x(y),
$$

and

$$
\Sigma_x=
\frac{1}{W_x}
\sum_y
w_x(y)(r_{xy}-\mu_x)(r_{xy}-\mu_x)^{\mathsf T}.
$$

Two neighborhood kernels are available:

- `hard`: equal weight inside a sphere of radius `covariance_radius`;
- `gaussian`: radial Gaussian weight. If `covariance_sigma` is omitted,
  $\sigma=R/\sqrt5$, matching the radial second moment of a uniform 3-D ball
  of radius $R$. The Gaussian is truncated at `truncate_sigma * sigma`.

### Directional covariance deficit

For a unit direction $d$,

$$
A_d(x)
=
1-
\sqrt{
\frac{d^{\mathsf T}\Sigma_xd}
{\operatorname{tr}\Sigma_x}
}.
$$

Small $A_d$ means that a large fraction of the local covariance is aligned
with $d$. If $\Sigma_x=\lambda vv^{\mathsf T}$ is rank one, then

$$
A_d(x)=1-|v^{\mathsf T}d|.
$$

### Principal-axis degeneracy

Let

$$
\lambda_1\ge\lambda_2\ge\lambda_3\ge0
$$

be the eigenvalues of $\Sigma_x$. Define

$$
S(x)=\frac{\lambda_2}{\lambda_1}.
$$

Values near zero indicate one strongly dominant covariance direction. Values
near one indicate that the two largest covariance directions are similar, so a
unique principal axis is less well defined.

If the local weighted covariance is degenerate, the implementation assigns
$A_d=S=1$.

### Frame invariance

If an orthogonal transformation $Q$ fixes the selected physical direction,
$Qd=d$, and the same physical neighborhood is expressed as

$$
r'_{xy}=Qr_{xy},
\qquad
\Sigma'_x=Q\Sigma_xQ^{\mathsf T},
$$

then

$$
C'_d=C_d,
\qquad
A'_d=A_d,
\qquad
S'=S.
$$

Therefore the filtration does not depend on how an orthonormal basis is chosen
inside the transverse plane $d^\perp$. This is a coordinate-frame statement;
it does not claim invariance to rotating and resampling a discrete voxel image.

### Usage

```python
import numpy as np
from directional_betti import frame_invariant_directional_betti

foreground = np.zeros((32, 32, 32), dtype=np.uint8)
foreground[6:26, 15:18, 15:18] = 1
foreground[15:18, 6:26, 15:18] = 1
roi = np.ones_like(foreground)

analysis = frame_invariant_directional_betti(
    foreground,
    roi,
    spacing=1.0,
    directions=("x", "y", "z"),
    covariance_radius=4.0,
    cone_radius=4.0,
    cone_half_extent=8.0,
    covariance_kernel="gaussian",
)

for direction, result in analysis.by_direction.items():
    print(direction, result.b0.shape)
```

`spacing`, `covariance_radius`, `cone_radius`, `cone_half_extent`, and
`covariance_sigma` must use the same physical unit. The present implementation
assumes isotropic voxel spacing.

Foreground voxels receive the filtration values above. Nonforeground top cubes
inside the ROI receive the terminal value `(1.25, 1.25, 1.25)`. Cubes outside
the ROI are absent.

## Command line

From an already constructed three-parameter `.npy` filtration:

```bash
directional-betti topology filtration.npy output_dir \
  --roi roi.npy \
  --grid-res 8
```

From a binary volume:

```bash
directional-betti invariant foreground.npy output_dir \
  --roi roi.npy \
  --spacing 1.0 \
  --covariance-radius 4.0 \
  --cone-radius 4.0 \
  --cone-half-extent 8.0 \
  --covariance-kernel gaussian \
  --grid-res 8
```

Use `--threads N` before the subcommand to set the Numba thread count.

## Source layout

The source is intentionally small:

```text
src/directional_betti/
    api.py          public Python API
    filtration.py   frame-invariant filtration
    geometry.py     physical neighborhood construction
    grid.py         threshold-grid discretization
    topology.py     exact cubical Betti construction
    events.py       cubical birth events
    roi.py          reusable cubical ROI connectivity
    graph.py        union-find graph sweeps
    io.py           optional CLI file input
    cli.py          command-line interface
```

The two scripts under `examples/` reproduce the two public use cases above.
