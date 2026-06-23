from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Single linear layer trained on top of frozen DiT features."""

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPProbe(nn.Module):
    """Small MLP trained on top of frozen DiT features."""

    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ================================================================================================
# KAN and Fourier KAN are more complex probe types which may be beneficial for our downstream task.


# KAN is ported from https://github.com/KindXiaoming/pykan.
# Fourier KAN follows https://github.com/abdalimran/FR-KAN-Text-Classification/blob/main/kan/kan_fourier.py.


def B_batch(
    x: torch.Tensor,
    grid: torch.Tensor,
    k: int = 0,
    extend: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.

    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100, 2)
    >>> grid = torch.linspace(-1, 1, steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    """

    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)

        value = (x - grid[:, :, : -(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, : -(k + 1)]) * B_km1[
            :, :, :-1
        ] + (grid[:, :, k + 1 :] - x) / (grid[:, :, k + 1 :] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]

    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value


def coef2curve(
    x_eval: torch.Tensor,
    grid: torch.Tensor,
    coef: torch.Tensor,
    k: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)

    """

    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum("ijk,jlk->ijl", b_splines, coef.to(b_splines.device))

    return y_eval


def curve2coef(
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    grid: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
        lamb : float
            regularized least square lambda

    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
    """
    # print('haha', x_eval.shape, y_eval.shape, grid.shape)
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    # print('mat', mat.shape)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    # print('y_eval', y_eval.shape)
    # coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except Exception:
        raise RuntimeError(
            f"Least squares failed with mat shape {mat.shape} and y_eval shape {y_eval.shape}. This may be due to insufficiently many samples or degenerate grid. Consider increasing the number of samples or adjusting the grid."
        )

    return coef


def extend_grid(grid: torch.Tensor, k_extend: int = 0) -> torch.Tensor:
    """
    extend grid
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid


def sparse_mask(in_dim: int, out_dim: int) -> torch.Tensor:
    """
    get sparse mask
    """
    in_coord = torch.arange(in_dim) * 1 / in_dim + 1 / (2 * in_dim)
    out_coord = torch.arange(out_dim) * 1 / out_dim + 1 / (2 * out_dim)

    dist_mat = torch.abs(out_coord[:, None] - in_coord[None, :])
    in_nearest = torch.argmin(dist_mat, dim=0)
    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1, 0)
    out_nearest = torch.argmin(dist_mat, dim=1)
    out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1, 0)
    all_connection = torch.cat([in_connection, out_connection], dim=0)
    mask = torch.zeros(in_dim, out_dim)
    mask[all_connection[:, 0], all_connection[:, 1]] = 1.0

    return mask


class KANProbe(nn.Module):
    """
    Spline KAN classifier head.

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        noise_scale: float
            spline scale at initialization
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base_mu: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        scale_base_sigma: float
            magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        scale_sp: float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            the id of activation functions that are locked
        device: str
            device
    """

    # TODO: thread other constructor params into config, likely through nested dataclass
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 2,
        grid_size: int = 5,
        k: int = 3,
        noise_scale: float = 0.5,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: nn.Module | None = None,
        grid_eps: float = 0.02,
        grid_range: Sequence[float] = (-1, 1),
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        device: str | torch.device = "cpu",
        sparse_init: bool = False,
    ) -> None:
        """'
        initialize a KANProbe

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            grid_size : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.

        Returns:
        --------
            self

        Example
        -------
        >>> model = KANProbe(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        """
        super().__init__()
        # size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.grid_size = grid_size
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=grid_size + 1)[None, :].expand(
            self.in_dim, grid_size + 1
        )
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (
            (torch.rand(self.grid_size + 1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / self.grid_size
        )
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))

        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        self.scale_base = torch.nn.Parameter(
            scale_base_mu * 1 / np.sqrt(in_dim)
            + scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(in_dim)
        ).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask
        ).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun if base_fun is not None else torch.nn.SiLU()

        self.grid_eps = grid_eps

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return classifier logits for input features.

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)

        Example
        -------
        >>> model = KANProbe(in_dim=3, out_dim=5)
        >>> x = torch.normal(0, 1, size=(100, 3))
        >>> y = model(x)
        >>> y.shape
        """
        base = self.base_fun(x)  # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y

        y = torch.sum(y, dim=1)
        return y

    def update_grid_from_samples(self, x: torch.Tensor, mode: str = "sample") -> None:
        """
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANProbe(in_dim=1, out_dim=1, grid_size=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3, 3, steps=100)[:, None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        """

        batch = x.shape[0]
        # x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval: int) -> torch.Tensor:
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            margin = 0.00
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) / num_interval
            grid_uniform = (
                grid_adaptive[:, [0]]
                - margin
                + h
                * torch.arange(
                    num_interval + 1,
                )[None, :].to(x.device)
            )
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        if mode == "grid":
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)

        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(
        self,
        parent: "KANProbe",
        x: torch.Tensor,
        mode: str = "sample",
    ) -> None:
        """
        update grid from a parent KANProbe & samples

        Args:
        -----
            parent : KANProbe
                a parent KANProbe (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = KANProbe(in_dim=1, out_dim=1, grid_size=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANProbe(in_dim=1, out_dim=1, grid_size=10, k=3)
        >>> x = torch.normal(0, 1, size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        """

        _ = x.shape[0]

        # shrink grid
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        """
        # based on samples
        def get_grid(num_interval: int) -> torch.Tensor:
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid"""

        # print('p', parent.grid)
        # based on interpolating parent grid
        def get_grid(num_interval: int) -> torch.Tensor:
            x_pos = parent.grid[:, parent.k : -parent.k]
            # print('x_pos', x_pos)
            sp2 = KANProbe(
                in_dim=1,
                out_dim=self.in_dim,
                k=1,
                grid_size=x_pos.shape[1] - 1,
                scale_base_mu=0.0,
                scale_base_sigma=0.0,
            ).to(x.device)

            # print('sp2_grid', sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim))
            # print('sp2_coef_shape', sp2.coef.shape)
            sp2_coef = curve2coef(
                sp2.grid[:, sp2.k : -sp2.k].permute(1, 0).expand(-1, self.in_dim),
                x_pos.permute(1, 0).unsqueeze(dim=2),
                sp2.grid[:, :],
                k=1,
            ).permute(1, 0, 2)
            _ = sp2_coef.shape
            # shp = sp2_coef.shape
            # sp2_coef = torch.cat([torch.zeros(shp[0], shp[1], 1), sp2_coef, torch.zeros(shp[0], shp[1], 1)], dim=2)
            # print('sp2_coef',sp2_coef)
            # print(sp2.coef.shape)
            sp2.coef.data = sp2_coef
            percentile = torch.linspace(-1, 1, self.grid_size + 1).to(self.grid.device)
            grid = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
            # print('c', grid)
            return grid

        grid = get_grid(num_interval)

        if mode == "grid":
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)

        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def get_subset(self, in_id: Sequence[int], out_id: Sequence[int]) -> "KANProbe":
        """
        get a smaller KANProbe from a larger KANProbe (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : KANProbe

        Example
        -------
        >>> kanlayer_large = KANProbe(in_dim=10, out_dim=10, grid_size=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0, 9], [1, 2, 3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        """
        spb = KANProbe(len(in_id), len(out_id), self.grid_size, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:, out_id]
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb


class FourierKANProbe(nn.Module):
    """Fourier KAN classifier head that returns logits directly."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        grid_size: int = 5,
        add_bias: bool = True,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.inputdim = feat_dim
        self.outdim = num_classes

        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, num_classes, feat_dim, grid_size) / (np.sqrt(feat_dim) * np.sqrt(grid_size))
        )
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(1, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        out_shape = x_shape[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(torch.arange(1, self.grid_size + 1, device=x.device), (1, 1, 1, self.grid_size))
        x_reshape = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        cos = torch.cos(k * x_reshape)
        sin = torch.sin(k * x_reshape)
        y = torch.sum(cos * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(sin * self.fouriercoeffs[1:2], (-2, -1))

        if self.add_bias:
            y += self.bias
        return torch.reshape(y, out_shape)
