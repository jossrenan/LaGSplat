"""
sim_collision.py — rollout du LNN avec collisions contre des obstacles image.

Depuis une frame de départ de la vidéo, on déroule le LNN deux fois, avec la même
condition initiale et le même forçage de pression :
  - libre : la dynamique apprise seule ;
  - avec contact : + Q(q, q̇) = Σ_k J_kᵀ f_k, force de pénalité des obstacles sur
    les centres des gaussiennes du décodeur (cf. collisions.py).
Les deux rollouts sont décodés côte à côte dans une vidéo MP4, obstacles tracés.

Obstacles en coordonnées image [0,1]² (x → droite, y → bas), option répétable :
    --obstacle 'disc:0.70,0.55,0.12'
    --obstacle 'plane:0.60,0,-1,0'       (mur vertical x = 0.6, libre à gauche)
    --obstacle 'box:0.65,0.40,1.0,0.60'
Sans --obstacle ni COLLISION_OBSTACLES : `--auto-obstacle` (défaut) place un mur
perpendiculaire à la plus grande excursion de l'objet dans le rollout libre, à
mi-course, pour que le contact ait lieu.

Lancer (interpréteur `py`, mêmes réglages que l'évaluation publiée) :
    py sim_collision.py --config ../cases/krauss2026_2seg_npz/config.py \
        --lnn lnn_lr1e-3_c1_s1_500ep_seed0.pt --video scr_2seg_32x32_val20.npz \
        --sigma 1 --sigma-pressure 1 --cq-eps 1 --obstacle 'disc:0.7,0.6,0.12'

Sortie : `collision[_<tag>].mp4` et `.npz` (trajectoires latentes) dans SAVE_DIR.
"""
import argparse

from _bootstrap import load_config
config = load_config()

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--obstacle', action='append', default=None,
                 help="obstacle 'plane:px,py,nx,ny' | 'disc:cx,cy,r' | 'box:x0,y0,x1,y1' "
                      '(répétable ; remplace COLLISION_OBSTACLES)')
_ap.add_argument('--no-auto-obstacle', action='store_true',
                 help="sans obstacle donné, ne PAS en placer un automatiquement")
_ap.add_argument('--stiffness', type=float, default=None, help='override COLLISION_STIFFNESS')
_ap.add_argument('--damping', type=float, default=None, help='override COLLISION_DAMPING')
_ap.add_argument('--friction', type=float, default=None, help='override COLLISION_FRICTION')
_ap.add_argument('--radius-scale', type=float, default=None, help='override COLLISION_RADIUS_SCALE')
_ap.add_argument('--substeps', type=int, default=None, help='override COLLISION_SUBSTEPS')
_ap.add_argument('--start-sec', type=float, default=10.0, help='frame de départ (s)')
_ap.add_argument('--horizon-sec', type=float, default=3.0, help='durée du rollout (s)')
_ap.add_argument('--no-pressure', action='store_true',
                 help='oscillation libre : pas de forçage de pression')
_ap.add_argument('--frames', type=int, default=None,
                 help='nb de frames chargées (défaut : start + horizon + 200)')
_ap.add_argument('--lnn', type=str, default='auto',
                 help="checkpoint LNN : 'auto' (lnn_rollout.pt sinon lnn.pt) | nom explicite")
_ap.add_argument('--decoder', type=str, default='auto',
                 help="checkpoint décodeur : 'auto' (decoder2dpt_ae.pt sinon decoder2dpt.pt)")
_ap.add_argument('--video', type=str, default=None,
                 help='nom de fichier vidéo (dans le dossier de VIDEO_DIR)')
_ap.add_argument('--sigma', type=float, default=None, help='override SMOOTH_LATENT_SIGMA')
_ap.add_argument('--sigma-pressure', type=float, default=None, help='override SMOOTH_PRESSURE_SIGMA')
_ap.add_argument('--cq-eps', type=float, default=None, help='override LNN_RAYLEIGH_CQ_EPS')
_ap.add_argument('--integrator', type=str, default=None,
                 choices=['verlet', 'rk4', 'gen_alpha'], help='override LNN_INTEGRATOR')
_ap.add_argument('--scale', type=int, default=8, help='agrandissement de la vidéo de sortie')
_ap.add_argument('--out', type=str, default=None, help='nom du MP4 de sortie')
_args, _ = _ap.parse_known_args()

if _args.integrator is not None:
    config.LNN_INTEGRATOR = _args.integrator
if _args.cq_eps is not None:
    config.LNN_RAYLEIGH_CQ_EPS = float(_args.cq_eps)
if _args.sigma is not None:
    config.SMOOTH_LATENT_SIGMA = float(_args.sigma)
if _args.sigma_pressure is not None:
    config.SMOOTH_PRESSURE_SIGMA = float(_args.sigma_pressure)

import numpy as np
import torch

from dataset import VideoFrameDataset
from models import LNN, build_encoder, WhitenedEncoder, load_latent_whiten
from models_2pt import build_decoder2pt
from viz import simulate_rk4, initial_velocity, get_sim_pressure
from train_lnn_fixedae import encode_all, smooth_latents, smooth_pressures
from collisions import GaussianContact, HalfPlane, Disc, Box, build_contact, parse_obstacle


def rollout(lnn, z0, v0, n_frames, pressure, collision, substeps):
    """Rollout de n_frames frames, `substeps` pas d'intégration par frame."""
    p = None if pressure is None else pressure.repeat_interleave(substeps, 0)
    with torch.no_grad():
        traj = simulate_rk4(lnn, z0, v0, n_steps=(n_frames - 1) * substeps + 1,
                            dt=1.0 / substeps, pressure=p, collision=collision)
    return traj[::substeps]                                        # (n_frames, d)


def auto_obstacle(contact_geom, traj):
    """
    Mur perpendiculaire à la plus grande excursion du centroïde de la partie
    MOBILE de l'objet (gaussiennes de ‖J_k‖ élevé), placé à mi-course vers
    l'extrémité atteinte : le rollout libre le traverse, le contact a lieu.
    """
    with torch.no_grad():
        Jn = contact_geom.J.flatten(1).norm(dim=1)                  # (K,)
        mobile = Jn >= torch.quantile(Jn, 0.8)
        x = contact_geom.centers(traj)[:, mobile]                   # (T, Km, 2)
        w = contact_geom.presence(traj)[:, mobile]                  # (T, Km)
        c = (w.unsqueeze(-1) * x).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-12)
        c0 = c[0]
        dev = c - c0                                                # (T, 2)
        i = dev.norm(dim=1).argmax()
        axis = dev[i] / dev[i].norm().clamp(min=1e-9)
        p = c0 + 0.5 * dev[i]
    return HalfPlane(p.tolist(), (-axis).tolist())                  # libre côté départ


def draw_obstacles(img, obstacles, color=(0, 0, 255)):
    """Trace le contour des obstacles sur une image BGR uint8 (H, W, 3)."""
    import cv2
    H, W = img.shape[:2]
    overlay = img.copy()
    for o in obstacles:
        if isinstance(o, Disc):
            c = (int(o.c[0] * W), int(o.c[1] * H))
            cv2.circle(overlay, c, int(o.r * W), color, -1, cv2.LINE_AA)
        elif isinstance(o, Box):
            lo, hi = (o.c - o.h).tolist(), (o.c + o.h).tolist()
            cv2.rectangle(overlay, (int(lo[0] * W), int(lo[1] * H)),
                          (int(hi[0] * W), int(hi[1] * H)), color, -1)
        else:   # générique : masque {φ < 0} évalué sur la grille
            ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W),
                                    indexing='ij')
            dev = next((v.device for v in vars(o).values() if torch.is_tensor(v)), 'cpu')
            mask = (o.sdf(torch.stack([xs, ys], -1).to(dev)) < 0).cpu().numpy()
            overlay[mask] = color
    return cv2.addWeighted(overlay, 0.45, img, 0.55, 0)


def main():
    import cv2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _use_pressure = getattr(config, 'LNN_PRESSURE', False) and not _args.no_pressure
    n_ch = 3 if config.ENC_COLOR else 1
    n_ch_dec = 3 if getattr(config, 'DEC_COLOR', True) else 1
    substeps = max(1, int(_args.substeps or getattr(config, 'COLLISION_SUBSTEPS', 4)))
    start = int(round(_args.start_sec / config.DT))
    horizon = max(2, int(round(_args.horizon_sec / config.DT)))
    n_load = _args.frames or start + horizon + 200

    video_dir = config.VIDEO_DIR if _args.video is None else config.VIDEO_DIR.parent / _args.video
    ds = VideoFrameDataset(
        video_dir=video_dir, img_size=config.IMG_SIZE, n_channels=n_ch,
        rest_video=config.REST_VIDEO, rest_n_frames=config.REST_N_FRAMES,
        rest_first_n_frames=getattr(config, 'REST_FIRST_N_FRAMES', 0),
        crop=getattr(config, 'CROP', None),
        load_pressure=_use_pressure, pressure_dir=getattr(config, 'PRESSURE_DIR', None),
        pressure_cols=getattr(config, 'PRESSURE_COLS', None),
        pressure_norm=getattr(config, 'PRESSURE_NORM', 101325.0), pressure_dt=config.DT,
        pressure_sync_offsets=getattr(config, 'PRESSURE_SYNC_OFFSETS', None),
        max_frames=n_load, store_uint8=True)
    n = len(ds.frames)
    assert n > start + 3, f'--start-sec trop grand : {n} frames chargées'

    # ── Encodeur + blanchiment, décodeur, LNN (comme eval_multistep_mse) ──────
    enc = build_encoder(config.IMG_SIZE, config.ENC_HIDDEN, config.LATENT_DIM,
                        n_channels=n_ch, normalize=getattr(config, 'ENC_NORMALIZE', False)).to(device)
    enc_path = config.SAVE_DIR / 'encoder_ae.pt'
    if not enc_path.exists():
        enc_path = config.SAVE_DIR / 'encoder.pt'
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    enc.eval()
    whiten = load_latent_whiten(config.SAVE_DIR, device, config.LATENT_DIM)
    if whiten is not None:
        enc = WhitenedEncoder(enc, whiten).to(device).eval()

    dec_path = config.SAVE_DIR / ('decoder2dpt_ae.pt' if _args.decoder == 'auto' else _args.decoder)
    if _args.decoder == 'auto' and not dec_path.exists():
        dec_path = config.SAVE_DIR / 'decoder2dpt.pt'
    dec_state = torch.load(dec_path, map_location=device)
    decoder = build_decoder2pt(
        latent_dim=config.LATENT_DIM, n_gaussians=dec_state['mu_raw'].shape[0],
        img_size=config.IMG_SIZE, n_channels=n_ch_dec).to(device)
    decoder.load_state_dict(dec_state)
    decoder.eval()

    lnn_path = config.SAVE_DIR / ('lnn_rollout.pt' if _args.lnn == 'auto' else _args.lnn)
    if _args.lnn == 'auto' and not lnn_path.exists():
        lnn_path = config.SAVE_DIR / 'lnn.pt'
    lnn = LNN(config.LATENT_DIM, config.LNN_HIDDEN).to(device)
    state = torch.load(lnn_path, map_location=device)
    try:
        lnn.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f'⚠ strict load échoué ({e}); repli strict=False.')
        lnn.load_state_dict(state, strict=False)
    lnn.eval()
    print(f'Encodeur {enc_path.name} | décodeur {dec_path.name} | LNN {lnn_path.name}')

    # ── Condition initiale et forçage ────────────────────────────────────────
    z = encode_all(enc, ds.frames, device)
    if getattr(config, 'SMOOTH_LATENT', False):
        z = smooth_latents(z.clone(), [n],
                           getattr(config, 'SMOOTH_LATENT_WINDOW', 13),
                           getattr(config, 'SMOOTH_LATENT_POLY', 3),
                           getattr(config, 'SMOOTH_LATENT_MODE', 'savgol'),
                           getattr(config, 'SMOOTH_LATENT_SIGMA', 10.0))
    if _use_pressure and getattr(config, 'SMOOTH_PRESSURE', False):
        ds.pressures = smooth_pressures(ds.pressures, ds.video_lengths)
    z = z.to(device)
    z0, v0 = z[start], initial_velocity(z[start:start + 3])
    pressure = get_sim_pressure(lnn, ds, start, horizon, device) if _use_pressure else None

    # ── Rollout libre, puis obstacles et rollout avec contact ────────────────
    traj_free = rollout(lnn, z0, v0, horizon, pressure, None, substeps)
    obstacles = _args.obstacle or list(getattr(config, 'COLLISION_OBSTACLES', []))
    obstacles = [parse_obstacle(o) for o in obstacles]
    if not obstacles:
        if _args.no_auto_obstacle:
            raise SystemExit("aucun obstacle : passer --obstacle 'disc:cx,cy,r' (cf. --help)")
        geom = GaussianContact(decoder, [], whiten=whiten).to(device)
        obstacles = [auto_obstacle(geom, traj_free)]
        print(f'Obstacle automatique : {obstacles[0]}')
    contact = build_contact(decoder, whiten=whiten, obstacles=obstacles,
                            stiffness=_args.stiffness, damping=_args.damping,
                            friction=_args.friction,
                            radius_scale=_args.radius_scale).to(device)
    print(contact)
    traj_col = rollout(lnn, z0, v0, horizon, pressure, contact, substeps)

    with torch.no_grad():
        pen_free = contact.penetration(traj_free)
        pen_col = contact.penetration(traj_col)
    px = config.IMG_SIZE[1]
    print(f'Pénétration max : libre {pen_free.max().item() * px:.2f} px  |  '
          f'avec contact {pen_col.max().item() * px:.2f} px  '
          f'(frames en contact : {(pen_col > 0).sum().item()}/{horizon})')
    if not torch.isfinite(traj_col).all():
        print('⚠ rollout avec contact non fini : baisser --stiffness ou monter --substeps.')

    # ── Décodage et vidéo côte à côte ─────────────────────────────────────────
    def decode(u):
        out = []
        with torch.no_grad():
            for i in range(0, len(u), 64):
                zb = u[i:i + 64]
                if whiten is not None:
                    zb = whiten.inverse(zb)
                out.append(decoder(zb).clamp(0, 1).cpu())
        return torch.cat(out)

    def to_bgr(x):
        a = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        a = np.repeat(a, 3, axis=2) if a.shape[2] == 1 else a[..., ::-1]
        H, W = a.shape[:2]
        return cv2.resize(np.ascontiguousarray(a), (W * _args.scale, H * _args.scale),
                          interpolation=cv2.INTER_NEAREST)

    img_free, img_col = decode(traj_free), decode(traj_col)
    tag = _args.out or 'collision.mp4'
    out = config.SAVE_DIR / tag
    H, W = config.IMG_SIZE[0] * _args.scale, config.IMG_SIZE[1] * _args.scale
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'),
                         round(1.0 / config.DT), (2 * W, H))
    for t in range(horizon):
        left = draw_obstacles(to_bgr(img_free[t]), obstacles, color=(160, 160, 160))
        right = draw_obstacles(to_bgr(img_col[t]), obstacles)
        frame = np.concatenate([left, right], axis=1)
        cv2.putText(frame, 'libre', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, 'contact', (W + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        vw.write(frame)
    vw.release()
    np.savez(out.with_suffix('.npz'), traj_free=traj_free.cpu().numpy(),
             traj_col=traj_col.cpu().numpy(), pen_free=pen_free.cpu().numpy(),
             pen_col=pen_col.cpu().numpy(), obstacles=np.array([repr(o) for o in obstacles]),
             start=start, dt=config.DT)
    print(f'Vidéo : {out}')


if __name__ == '__main__':
    main()
