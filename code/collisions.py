"""
collisions.py — contact entre l'objet filmé et des obstacles placés dans l'image.

Principe (le même que pour une poussée, cf. README) : les primitives du décodeur
GS sont des points explicites qui bougent avec l'objet. Pour la gaussienne k, la
coupe conditionnelle à l'état latent z donne le centre

    μ_k(z) = μ_xy,k + Σ_xz,k Σ_zz,k⁻¹ (z − μ_z,k)          (Schur, models_2pt)

AFFINE en z : son Jacobien J_k = Σ_xz,k Σ_zz,k⁻¹ (2×d) est exact et constant. Le
LNN vit dans l'espace blanchi u (z = W_invᵀ u + mean) ⟹ J_k,u = J_k W_invᵀ.

Chaque obstacle est un solide rigide et fixe, décrit par une distance signée
φ(x) (> 0 dans l'espace libre) en coordonnées image normalisées [0,1]² (x vers la
droite, y vers le bas). La gaussienne k, de rayon r_k (r_k = RADIUS_SCALE·σ_max
de sa covariance conditionnelle), pénètre de δ_k = max(0, r_k − φ(μ_k)). On lui
applique une force de pénalité de Kelvin-Voigt, normale n_k = ∇φ :

    f_n,k = max(0, k_c δ_k − c_c (n_k·ẋ_k)) ,   ẋ_k = J_k,u u̇
    f_t,k = −μ f_n,k ẋ_t,k / √(‖ẋ_t,k‖² + ε²)    (Coulomb régularisé, optionnel)

pondérée par la présence w_k = α_k·w_z,k(u) / Σ_j α_j·w_z,j(u) (une gaussienne
éteinte à cet état ne touche rien ; la normalisation rend k_c indépendant du
nombre de gaussiennes). La force généralisée qui entre dans les équations de
Lagrange est le pull-back

    Q(u, u̇) = Σ_k J_k,uᵀ w_k (f_n,k n_k + f_t,k) .

Les gaussiennes du décor (fond fixe) ont J_k ≈ 0 et ne transmettent rien : seule
la partie mobile de l'objet est repoussée par les obstacles.

Q est ajoutée au second membre AVANT la résolution par la masse (cf.
viz.simulate_rk4 et LNN.accel(f_ext=…)). C'est une pénalité explicite : garder
ω_c·dt < 2 avec ω_c ~ √(k_c·‖J‖²/m) (augmenter les sous-pas si le contact
« traverse » ou explose).

Syntaxe des obstacles (CLI / config.COLLISION_OBSTACLES) :
    'plane:px,py,nx,ny'     demi-plan solide ; n pointe vers l'espace LIBRE
    'disc:cx,cy,r'          disque solide de centre (cx, cy) et rayon r
    'box:x0,y0,x1,y1'       rectangle solide aligné sur les axes
"""
import torch
import torch.nn as nn


# ── Obstacles (distance signée, > 0 dans l'espace libre) ─────────────────────

class Obstacle:
    """Solide fixe défini par sa distance signée φ(x), x : (..., 2) → (...)."""

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """n = ∇φ / ‖∇φ‖ : (..., 2), normale sortante (vers l'espace libre)."""
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            g = torch.autograd.grad(self.sdf(xg).sum(), xg)[0]
        return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def to(self, device):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self


class HalfPlane(Obstacle):
    """Demi-plan solide {x : n·(x − p) < 0} ; n pointe vers l'espace libre."""

    def __init__(self, point, normal):
        n = torch.as_tensor(normal, dtype=torch.float32)
        self.n = n / n.norm().clamp(min=1e-9)
        self.p = torch.as_tensor(point, dtype=torch.float32)

    def sdf(self, x):
        return ((x - self.p) * self.n).sum(-1)

    def normal(self, x):
        return self.n.expand_as(x)

    def __repr__(self):
        return f'HalfPlane(p={self.p.tolist()}, n={self.n.tolist()})'


class Disc(Obstacle):
    """Disque solide de centre c et rayon R."""

    def __init__(self, center, radius):
        self.c = torch.as_tensor(center, dtype=torch.float32)
        self.r = float(radius)

    def sdf(self, x):
        return (x - self.c).norm(dim=-1) - self.r

    def normal(self, x):
        d = x - self.c
        return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def __repr__(self):
        return f'Disc(c={self.c.tolist()}, r={self.r})'


class Box(Obstacle):
    """Rectangle solide [x0, x1] × [y0, y1]."""

    def __init__(self, x0, y0, x1, y1):
        lo = torch.tensor([min(x0, x1), min(y0, y1)], dtype=torch.float32)
        hi = torch.tensor([max(x0, x1), max(y0, y1)], dtype=torch.float32)
        self.c = 0.5 * (lo + hi)
        self.h = 0.5 * (hi - lo)

    def sdf(self, x):
        q = (x - self.c).abs() - self.h
        outside = q.clamp(min=0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0)
        return outside + inside

    def __repr__(self):
        lo, hi = self.c - self.h, self.c + self.h
        return f'Box({lo.tolist()} → {hi.tolist()})'


def parse_obstacle(spec) -> Obstacle:
    """'disc:0.7,0.5,0.1' | 'plane:0,0.8,0,-1' | 'box:0.6,0.2,0.9,0.4' → Obstacle."""
    if isinstance(spec, Obstacle):
        return spec
    kind, _, args = str(spec).partition(':')
    vals = [float(a) for a in args.split(',') if a.strip()]
    kind = kind.strip().lower()
    if kind in ('plane', 'halfplane', 'wall') and len(vals) == 4:
        return HalfPlane(vals[:2], vals[2:])
    if kind in ('disc', 'disk', 'circle') and len(vals) == 3:
        return Disc(vals[:2], vals[2])
    if kind in ('box', 'rect') and len(vals) == 4:
        return Box(*vals)
    raise ValueError(
        f"obstacle invalide : {spec!r} (attendu 'plane:px,py,nx,ny', "
        f"'disc:cx,cy,r' ou 'box:x0,y0,x1,y1')")


# ── Modèle de contact des gaussiennes ────────────────────────────────────────

class GaussianContact(nn.Module):
    """
    Force généralisée de contact Q(u, u̇) des gaussiennes du décodeur contre des
    obstacles image. Géométrie du décodeur GELÉE à la construction.

    decoder   : GaussianSplatDecoder2pt (ou sa variante gsplat — même géométrie)
    obstacles : liste d'Obstacle ou de chaînes (cf. parse_obstacle)
    whiten    : LatentWhiten ou None — le LNN vit dans l'espace blanchi u
    stiffness : k_c, raideur de pénalité (accélération image par unité de
                pénétration, pour un objet de présence totale 1)
    damping   : c_c, amortissement normal (ne tire jamais : f_n ≥ 0)
    friction  : μ, coefficient de Coulomb régularisé (0 = glissement parfait)
    radius_scale : r_k = radius_scale · σ_max(Σ_cond,k) (0 = centres ponctuels)
    min_presence : gaussiennes de α_k sous ce seuil ignorées (fond transparent)
    """

    def __init__(self, decoder, obstacles, whiten=None, stiffness=2e3,
                 damping=20.0, friction=0.0, radius_scale=1.0,
                 min_presence=1e-3, eps_t=1e-4):
        super().__init__()
        self.obstacles = [parse_obstacle(o) for o in obstacles]
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.friction = float(friction)
        self.eps_t = float(eps_t)

        with torch.no_grad():
            mu_xy, mu_z, Sigma, alpha, _ = decoder._decode_params()
            d = mu_z.shape[1]
            Sxx, Sxz, Szz = Sigma[:, :2, :2], Sigma[:, :2, 2:], Sigma[:, 2:, 2:]
            eye = torch.eye(d, device=Szz.device, dtype=Szz.dtype)
            Szz_inv = torch.linalg.inv(Szz + 1e-4 * eye)      # même régularisation que forward
            J = Sxz @ Szz_inv                                 # (K, 2, d)  ∂μ/∂z
            S_cond = Sxx - J @ Sxz.transpose(-1, -2)          # (K, 2, 2)
            radius = radius_scale * torch.linalg.eigvalsh(S_cond)[:, -1].clamp(min=0).sqrt()

            keep = alpha > min_presence
            # Espace blanchi : z = u @ W_inv + mean ⟹ z − μ_z = (u − μ_z,u) @ W_inv
            if whiten is not None:
                W, W_inv, mean = whiten.W.float(), whiten.W_inv.float(), whiten.mean.float()
                J_u = J @ W_inv.t()                               # (K, 2, d)
                mu_z_u = (mu_z - mean) @ W                        # (K, d)
                Szzi_u = W_inv @ Szz_inv @ W_inv.t()              # (K, d, d)
            else:
                J_u, mu_z_u, Szzi_u = J, mu_z, Szz_inv

        self.register_buffer('mu_xy', mu_xy[keep].detach().clone())
        self.register_buffer('mu_z', mu_z_u[keep].detach().clone())
        self.register_buffer('J', J_u[keep].detach().clone())
        self.register_buffer('Szzi', Szzi_u[keep].detach().clone())
        self.register_buffer('alpha', alpha[keep].detach().clone())
        self.register_buffer('radius', radius[keep].detach().clone())
        for o in self.obstacles:
            o.to(self.mu_xy.device)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        for o in self.obstacles:
            o.to(self.mu_xy.device)
        return out

    # ── Géométrie ────────────────────────────────────────────────────────────

    def centers(self, u: torch.Tensor) -> torch.Tensor:
        """u : (B, d) → centres image μ_k(u) : (B, K, 2)."""
        du = u.unsqueeze(1) - self.mu_z.unsqueeze(0)               # (B, K, d)
        return self.mu_xy.unsqueeze(0) + torch.einsum('kpd,bkd->bkp', self.J, du)

    def presence(self, u: torch.Tensor) -> torch.Tensor:
        """u : (B, d) → poids normalisés w_k(u) = α_k w_z,k / Σ_j α_j w_z,j : (B, K)."""
        du = u.unsqueeze(1) - self.mu_z.unsqueeze(0)               # (B, K, d)
        quad = torch.einsum('bkd,kde,bke->bk', du, self.Szzi, du).clamp(max=40.0)
        w = self.alpha.unsqueeze(0) * torch.exp(-0.5 * quad)
        return w / w.sum(-1, keepdim=True).clamp(min=1e-12)

    def penetration(self, u: torch.Tensor) -> torch.Tensor:
        """u : (B, d) → pénétration maximale (pondérée par présence > 1e-4) : (B,)."""
        x, w = self.centers(u), self.presence(u)
        pen = torch.zeros_like(w)
        for o in self.obstacles:
            pen = torch.maximum(pen, (self.radius - o.sdf(x)).clamp(min=0))
        return (pen * (w > 1e-4 / w.shape[-1])).max(-1).values

    # ── Force ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Force généralisée de contact Q(u, u̇).
        u, v : (B, d) ou (d,)  → Q : même forme.
        """
        squeeze = u.dim() == 1
        if squeeze:
            u, v = u.unsqueeze(0), v.unsqueeze(0)
        u, v = u.detach().float(), v.detach().float()
        x = self.centers(u)                                         # (B, K, 2)
        xdot = torch.einsum('kpd,bd->bkp', self.J, v)               # (B, K, 2)
        w = self.presence(u)                                        # (B, K)

        f = torch.zeros_like(x)                                     # (B, K, 2)
        for o in self.obstacles:
            pen = (self.radius - o.sdf(x)).clamp(min=0)             # (B, K)
            touching = pen > 0
            if not touching.any():
                continue
            n = o.normal(x)                                         # (B, K, 2)
            vn = (xdot * n).sum(-1)                                 # (B, K)
            fn = (self.stiffness * pen - self.damping * vn).clamp(min=0) * touching
            f = f + fn.unsqueeze(-1) * n
            if self.friction > 0:
                vt = xdot - vn.unsqueeze(-1) * n
                vt_norm = (vt.pow(2).sum(-1, keepdim=True) + self.eps_t ** 2).sqrt()
                f = f - (self.friction * fn).unsqueeze(-1) * vt / vt_norm

        Q = torch.einsum('kpd,bkp->bd', self.J, w.unsqueeze(-1) * f)  # Σ J_kᵀ w_k f_k
        return Q.squeeze(0) if squeeze else Q

    def extra_repr(self):
        return (f'K={self.J.shape[0]}, obstacles={self.obstacles}, '
                f'k={self.stiffness:g}, c={self.damping:g}, mu={self.friction:g}')


def build_contact(decoder, whiten=None, obstacles=None, cfg=None, **overrides):
    """Construit un GaussianContact à partir des clés COLLISION_* de la config."""
    if cfg is None:
        import config as cfg
    obstacles = obstacles if obstacles is not None else getattr(cfg, 'COLLISION_OBSTACLES', [])
    if not obstacles:
        return None
    kw = dict(
        stiffness=getattr(cfg, 'COLLISION_STIFFNESS', 2e3),
        damping=getattr(cfg, 'COLLISION_DAMPING', 20.0),
        friction=getattr(cfg, 'COLLISION_FRICTION', 0.0),
        radius_scale=getattr(cfg, 'COLLISION_RADIUS_SCALE', 1.0),
        min_presence=getattr(cfg, 'COLLISION_MIN_PRESENCE', 1e-3),
    )
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return GaussianContact(decoder, obstacles, whiten=whiten, **kw)
