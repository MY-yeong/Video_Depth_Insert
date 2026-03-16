import os, math, glob
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "2")
DRAW_POSE_DOT_ONLY = True   

print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
import numpy as np
import cv2
from tqdm import tqdm
import torch
from scipy.signal import savgol_filter

VIDEO_PATH         = '/home/cvlab/mykim/Video/Test/VPP/0102/desk_0102.mp4'   #orl video
OUT_ROOT           = '/home/cvlab/mykim/Video/Test/VPP/0102/result/2method'     #output dir
DEPTH_DIR          = '/home/cvlab/mykim/Video/Test/VPP/0102/Depth_map/desk_0102_depths_npy'   #depth dir

DO_COMPUTE_FLOW    = True    
DO_ESTIMATE_POSE   = True      
FOV_DEG            = 60.0        
MAX_DIM            = 512          
DEVICE             = 'cuda'

ASSET_DIR          = '/home/cvlab/mykim/Video/Test/VPP/data/blender_result/1_nobg'    
ASSET_MASK_DIR     = None          
ORDER_SIGN         = +1            
FIRST_INSERT_FRAME = 0          

# Insertion position
ANCHOR_XY          = (1150, 600)   
DRAW_DEBUG_TEXT    = False

# yaw Smoothing
DO_SG_SMOOTH       = True
SG_WIN             = 11            
SG_POLY            = 2


USE_H1_EMA_COLOR    = True     
H1_RING_DILATE      = 15       
H1_GAIN_CLIP        = (0.9, 1.1)   # RGB
H1_L_A_CLIP         = (0.9, 1.1)   # LAB a 
H1_L_B_CLIP         = (-8.0, 8.0)  # LAB b 
H1_EMA_MOMENTUM     = 0.85     # EMA

USE_B1_MULTIBAND    = False    
B1_LEVELS           = 3        

EXPORT_SRC_MASK     = True     
EXPORT_FPS_OVERRIDE = None      


def ensure_dir(d): os.makedirs(d, exist_ok=True)

def list_sorted_images(folder, exts=(".png",".jpg",".jpeg",".webp")):
    if not folder or not os.path.isdir(folder): return []
    fs = []
    for e in exts: fs += glob.glob(os.path.join(folder, f"*{e}"))
    return sorted(fs)

def alpha_from_rgba_or_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3 and im.shape[2] == 4:
        bgr = im[..., :3]
        a   = im[..., 3]
        m   = a.astype(np.uint8)
        return bgr, m
    if im.ndim == 2:
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        g = im
    else:
        bgr = im
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = (g > 0).astype(np.uint8) * 255
    return bgr, m

class EmaColorMatcher:

    def __init__(self, gain_clip=(0.9,1.1), l_a_clip=(0.9,1.1), l_b_clip=(-8,8), momentum=0.85):
        self.gain_clip = gain_clip
        self.l_a_clip  = l_a_clip
        self.l_b_clip  = l_b_clip
        self.m         = float(momentum)
        self.a = None   
        self.b = None  
        self.g = None  

    def _ema(self, prev, cur):
        if prev is None: return cur
        return self.m * prev + (1.0 - self.m) * cur

    def apply(self, obj_bgr, roi_bgr, mask_u8, ring_dilate=15):
        se   = np.ones((ring_dilate, ring_dilate), np.uint8)
        dil  = cv2.dilate(mask_u8, se, iterations=1)
        ring = cv2.bitwise_and(dil, cv2.bitwise_not(mask_u8))
        m_obj  = (mask_u8 > 0)
        m_ring = (ring    > 0)

        if m_obj.sum() < 50 or m_ring.sum() < 50:
            return obj_bgr

        # LAB Luminosity Affine
        obj_lab = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_obj_mean = float(np.median(obj_lab[...,0][m_obj]))
        L_bg_mean  = float(np.median(roi_lab[...,0][m_ring]))
        a_hat = np.clip((L_bg_mean / (L_obj_mean + 1e-6)), self.l_a_clip[0], self.l_a_clip[1])
        b_hat = 0.0
        b_hat = np.clip(b_hat, self.l_b_clip[0], self.l_b_clip[1])
        self.a = self._ema(self.a, a_hat)
        self.b = self._ema(self.b, b_hat)
        obj_lab[...,0] = np.clip(self.a * obj_lab[...,0] + self.b, 0, 255)
        obj_adj = cv2.cvtColor(obj_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


        mean_obj = np.median(obj_adj[m_obj].reshape(-1,3), axis=0)
        mean_bg  = np.median(roi_bgr[m_ring].reshape(-1,3), axis=0)
        g_hat    = np.clip(mean_bg / (mean_obj + 1e-6), self.gain_clip[0], self.gain_clip[1])
        if self.g is None: self.g = g_hat
        else:              self.g = self.m * self.g + (1.0 - self.m) * g_hat
        obj_adj = np.clip(obj_adj.astype(np.float32) * self.g.reshape(1,1,3), 0, 255).astype(np.uint8)
        return obj_adj

def _pyr_down(img):
    return cv2.pyrDown(img)

def _pyr_up(img, size):
    up = cv2.pyrUp(img)
    if up.shape[:2] != size:
        up = cv2.resize(up, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return up

def multiband_blend(bg_roi, fg_roi, alpha01, levels=3):
    
    L = max(1, int(levels))
    bg = bg_roi.astype(np.float32) / 255.0
    fg = fg_roi.astype(np.float32) / 255.0
    a  = alpha01.astype(np.float32)
    a3 = np.dstack([a,a,a]) if a.ndim == 2 else a

    
    G_bg, G_fg, G_a = [bg], [fg], [a3]
    for _ in range(L-1):
        G_bg.append(_pyr_down(G_bg[-1]))
        G_fg.append(_pyr_down(G_fg[-1]))
        G_a .append(_pyr_down(G_a [-1]))

    
    L_bg, L_fg = [], []
    for i in range(L-1):
        size = G_bg[i].shape[:2]
        L_bg.append(G_bg[i] - _pyr_up(G_bg[i+1], size))
        L_fg.append(G_fg[i] - _pyr_up(G_fg[i+1], size))
    L_bg.append(G_bg[-1]); L_fg.append(G_fg[-1])

    L_blend = []
    for i in range(L):
        Li = L_fg[i] * G_a[i] + L_bg[i] * (1.0 - G_a[i])
        L_blend.append(Li)

    out = L_blend[-1]
    for i in range(L-2, -1, -1):
        size = L_blend[i].shape[:2]
        out = _pyr_up(out, size) + L_blend[i]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def overlay_center(frame, obj_bgr, obj_mask, center_xy, matcher=None):

    H, W = frame.shape[:2]
    h, w = obj_bgr.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    x0 = np.clip(cx - w//2, 0, W-1); x1 = np.clip(x0 + w, 0, W)
    y0 = np.clip(cy - h//2, 0, H-1); y1 = np.clip(y0 + h, 0, H)

    obj_c = obj_bgr[0:y1-y0, 0:x1-x0]
    msk_c = obj_mask[0:y1-y0, 0:x1-x0]
    if msk_c.size == 0 or obj_c.size == 0:
        return frame

    roi = frame[y0:y1, x0:x1].astype(np.uint8)
    alpha01 = (msk_c.astype(np.float32) / 255.0)


    obj_m = obj_c

    if USE_H1_EMA_COLOR and matcher is not None:
        try:
            obj_m = matcher.apply(obj_m, roi, msk_c, ring_dilate=H1_RING_DILATE)
        except Exception:
            pass


    if USE_B1_MULTIBAND:
        out_roi = multiband_blend(roi, obj_m, alpha01, levels=B1_LEVELS)
    else:
        a3 = alpha01[..., None] if alpha01.ndim == 2 else alpha01
        out_roi = (obj_m.astype(np.float32) * a3 +
                roi.astype(np.float32)    * (1.0 - a3))
        out_roi = np.clip(out_roi, 0, 255).astype(np.uint8)

    frame[y0:y1, x0:x1] = out_roi
    return frame

class Assets360:
    def __init__(self, rgb_dir, mask_dir=None, start_index_as_zero=0, order_sign=+1):
        self.paths = list_sorted_images(rgb_dir)
        assert len(self.paths) > 0, f"No assets in {rgb_dir}"
        self.N = len(self.paths)
        self.base_index = int(start_index_as_zero) % self.N
        self.order_sign = +1 if order_sign >= 0 else -1

        b0, m0 = alpha_from_rgba_or_rgb(self.paths[0])
        H0, W0 = b0.shape[:2]
        
        b0 = ((b0.astype(np.float32) * (m0[..., None].astype(np.float32) / 255.0))).astype(np.uint8)

        self.H, self.W = H0, W0
        self.bgrs = [b0]
        self.masks= [m0]

        for p in self.paths[1:]:
            b, m = alpha_from_rgba_or_rgb(p)
            if b.shape[:2] != (H0, W0):
                b = cv2.resize(b, (W0, H0), interpolation=cv2.INTER_AREA)
                m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_LINEAR)
            b = ((b.astype(np.float32) * (m[..., None].astype(np.float32) / 255.0))).astype(np.uint8)
            self.bgrs.append(b)
            self.masks.append(m)

        if mask_dir:
            mps = list_sorted_images(mask_dir)
            assert len(mps) == self.N, "The number of masks must match the number of assets."
            ms = []
            for p in mps:
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m.shape[:2] != (H0, W0):
                    m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_LINEAR)
                ms.append(m)
            self.masks = ms

        self._prev_pos = None

    def _blend(self, k0, k1, t):
        b0, m0 = self.bgrs[k0%self.N], self.masks[k0%self.N]
        b1, m1 = self.bgrs[k1%self.N], self.masks[k1%self.N]
        if t <= 0: return b0, m0
        if t >= 1: return b1, m1
        b = ((1-t) * b0.astype(np.float32) + t * b1.astype(np.float32)).astype(np.uint8)
        m = ((1-t) * m0.astype(np.float32) + t * m1.astype(np.float32)).astype(np.uint8)
        return b, m

    def sample_by_angle(self, angle_deg_mod360):
        a = float(angle_deg_mod360) % 360.0
        pos = self.base_index + self.order_sign * (a / 360.0) * self.N   
        self._prev_pos = pos

        k0 = int(np.floor(pos))
        k1 = k0 + 1
        t  = pos - np.floor(pos)
        b, m = self._blend(k0, k1, t)
        return b, m, pos, k0%self.N, k1%self.N, t


class RAFTFlow:
    def __init__(self, device='cuda'):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        self.device = torch.device('cuda' if (device=='cuda' and torch.cuda.is_available()) else 'cpu')
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(self.device).eval()
    @torch.no_grad()
    def infer_pair(self, img1_bgr, img2_bgr, max_dim=512):
        im1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        im2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        h, w = im1.shape[:2]
        t1 = torch.from_numpy(im1).permute(2,0,1).unsqueeze(0).to(self.device)
        t2 = torch.from_numpy(im2).permute(2,0,1).unsqueeze(0).to(self.device)
        flows = self.model(t1, t2)
        flow = flows[-1][0].permute(1,2,0).detach().cpu().numpy()
        return flow, (h, w)

def upsample_flow_to(flow_m, src_hw, dst_hw):
    h_m, w_m = src_hw
    H, W     = dst_hw
    fx = W / w_m; fy = H / h_m
    flow_rs = cv2.resize(flow_m, (W, H), interpolation=cv2.INTER_LINEAR)
    flow_rs[..., 0] *= fx
    flow_rs[..., 1] *= fy
    return flow_rs



def load_depths_from_dir(depth_dir, T, H, W):
    files = [os.path.join(depth_dir, f"{i:05d}.npy") for i in range(T)]
    depths = []
    for fp in files:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"[Depth] no find：{fp}")
        d = np.load(fp).astype(np.float32)
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        depths.append(d)
    return depths

def build_K_from_fov(width, height, fov_deg=60.0): 
    fx = fy = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
    cx, cy = width/2.0, height/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)

def backproject(depth, K, xs, ys):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    Z = depth[ys, xs]
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=1)

def umeyama(P, Q):
    muP, muQ = P.mean(0), Q.mean(0)
    X, Y = P-muP, Q-muQ
    U,S,Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = muQ - R @ muP
    return R, t

def estimate_global_poses(depths, flows_fwd, K, grid_step=16):
    T = len(depths)
    H, W = depths[0].shape
    Rs, ts = [np.eye(3)], [np.zeros(3)]
    R_cum = np.eye(3); t_cum = np.zeros(3)

    ys = np.arange(grid_step//2, H, grid_step)
    xs = np.arange(grid_step//2, W, grid_step)
    XS, YS = np.meshgrid(xs, ys)
    xs0 = XS.reshape(-1).astype(np.int32)
    ys0 = YS.reshape(-1).astype(np.int32)

    for i in tqdm(range(T-1), desc="Pose"):
        f01 = flows_fwd[i]
        x1s = xs0 + f01[ys0, xs0, 0]
        y1s = ys0 + f01[ys0, xs0, 1]
        inb = (x1s>=0)&(x1s<W)&(y1s>=0)&(y1s<H)
        xs_i = xs0[inb]; ys_i = ys0[inb]
        x1s  = x1s[inb].astype(np.int32); y1s = y1s[inb].astype(np.int32)

        z0 = depths[i][ys_i, xs_i]
        z1 = depths[i+1][y1s, x1s]
        valid = np.isfinite(z0)&np.isfinite(z1)&(z0>1e-6)&(z1>1e-6)
        xs_i, ys_i, x1s, y1s = xs_i[valid], ys_i[valid], x1s[valid], y1s[valid]
        if len(xs_i) < 50:
            Rs.append(R_cum.copy()); ts.append(t_cum.copy()); continue

        P0 = backproject(depths[i],   K, xs_i, ys_i)
        P1 = backproject(depths[i+1], K, x1s,  y1s)
        R_rel, t_rel = umeyama(P0, P1)
        R_cum = R_rel @ R_cum
        t_cum = R_rel @ t_cum + t_rel

        Rs.append(R_cum.copy()); ts.append(t_cum.copy())
    return Rs, ts

def euler_from_Rs(Rs):
    yaws, pitches, rolls = [], [], []
    for R in Rs:
        yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
        pitch = math.degrees(math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2)))
        roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
        yaws.append(yaw); pitches.append(pitch); rolls.append(roll)
    return np.asarray(yaws), np.asarray(pitches), np.asarray(rolls)


def main():
    ensure_dir(OUT_ROOT)

    # load video 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise FileNotFoundError(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    H0, W0 = 1080, 1920
    print(f"[info] video: {W0}x{H0} @ {fps:.2f}fps, frames={len(frames)}")

    # 
    depths = load_depths_from_dir(DEPTH_DIR, len(frames), H0, W0)

    # optical flow
    flows_f = [None]*(len(frames)-1)
    if DO_COMPUTE_FLOW:
        raft = RAFTFlow(DEVICE)
        for i in tqdm(range(len(frames)-1), desc="Flow"):
            f_m, hw_m = raft.infer_pair(frames[i], frames[i+1], MAX_DIM)
            flows_f[i] = upsample_flow_to(f_m, hw_m, (H0, W0))

    if DO_ESTIMATE_POSE:
        K = build_K_from_fov(W0, H0, FOV_DEG)
        Rs, ts = estimate_global_poses(depths, flows_f, K, grid_step=16) 
        
        u0, v0 = 1150, 600
        z0 = float(depths[0][v0, u0]) # 행 = 세로 = y = v , 열 = 가로 = x = u
        X0 = backproject(depths[0], K, np.array([u0], np.int32), np.array([v0], np.int32))[0] 
        
        X0_col = X0.reshape(3, 1)  # (3,1) 

        Xis = []  
        for i in range(len(frames)):
            R_i = Rs[i]            # (3,3)
            t_i = ts[i].reshape(3, 1)  # (3,1)

            X_i = (R_i @ X0_col + t_i).reshape(3)  # (3,)
            Xis.append(X_i)

        Xis = np.stack(Xis, axis=0)  # (T,3)
        print("[Step3] Xis shape =", Xis.shape)
        print("[Step3] X0 =", X0, " -> X1 =", Xis[1])

      
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        eps = 1e-6

        # u = fx * X/Z + cx,  v = fy * Y/Z + cy
        u = fx * (Xis[:, 0] / (Xis[:, 2] + eps)) + cx
        v = fy * (Xis[:, 1] / (Xis[:, 2] + eps)) + cy

        anchors_pose = np.stack([u, v], axis=1).astype(np.float32)  # (T,2)
        print("[Step4] anchors_pose[0] =", anchors_pose[0], "anchors_pose[1] =", anchors_pose[1])



        yaw_mod = np.mod(euler_from_Rs(Rs)[0], 360.0).astype(np.float32)
    else:
        yaw_mod = np.zeros(len(frames), dtype=np.float32)

    yaw_unwrapped = np.unwrap(np.deg2rad(yaw_mod))
    yaw_unwrapped = np.rad2deg(yaw_unwrapped).astype(np.float32)

    # align
    start_yaw_cont = yaw_unwrapped[FIRST_INSERT_FRAME]
    rel_yaw_cont   = yaw_unwrapped - start_yaw_cont     
    obj_angle_cont = -rel_yaw_cont                      

    # smoothing
    if DO_SG_SMOOTH and len(obj_angle_cont) >= SG_WIN:
        obj_angle_cont = savgol_filter(obj_angle_cont, window_length=SG_WIN,
                                       polyorder=SG_POLY, mode='interp')

    # reflect [0,360)
    obj_angle_mod = np.mod(obj_angle_cont, 360.0).astype(np.float32)


    ax, ay = ANCHOR_XY
    anchors = [(float(ax), float(ay))]
    for i in range(len(frames)-1):
        f = flows_f[i]
        if f is None:
            anchors.append((ax, ay)); continue
        x0 = int(round(ax - 100)); x1 = int(round(ax + 100))
        y0 = int(round(ay - 100)); y1 = int(round(ay + 100))
        x0, y0 = max(0,x0), max(0,y0)
        x1, y1 = min(W0-1,x1), min(H0-1,y1)
        patch = f[y0:y1, x0:x1, :]
        dx = float(np.median(patch[...,0])) if patch.size else 0.0
        dy = float(np.median(patch[...,1])) if patch.size else 0.0
        ax += dx; ay += dy
        anchors.append((ax, ay))
    anchors = np.array(anchors, dtype=np.float32)

    # assets
    assets = Assets360(ASSET_DIR, ASSET_MASK_DIR,
                       order_sign=ORDER_SIGN)

    # output video
    ensure_dir(OUT_ROOT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUT_ROOT, 'final_blended.mp4')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W0, H0))

    pose_path = os.path.join(OUT_ROOT, 'pose_up_down.mp4')
    vw_pose = cv2.VideoWriter(pose_path, fourcc, fps, (W0, H0))

    flow_path = os.path.join(OUT_ROOT, 'flow_up_down.mp4')  
    vw_flow = cv2.VideoWriter(flow_path, fourcc, fps, (W0, H0))



    if EXPORT_SRC_MASK:
        src_path = os.path.join(OUT_ROOT, 'insert_source.mp4')
        msk_path = os.path.join(OUT_ROOT, 'insert_mask.mp4')
        fps2 = float(EXPORT_FPS_OVERRIDE) if EXPORT_FPS_OVERRIDE else float(fps)
        vsrc = cv2.VideoWriter(src_path, fourcc, fps2, (W0, H0))
        vmsk = cv2.VideoWriter(msk_path, fourcc, fps2, (W0, H0))
    else:
        vsrc = vmsk = None

    matcher = EmaColorMatcher(
        gain_clip=H1_GAIN_CLIP,
        l_a_clip=H1_L_A_CLIP,
        l_b_clip=H1_L_B_CLIP,
        momentum=H1_EMA_MOMENTUM
    )

    print("[info] start blending ...")
    for i in tqdm(range(len(frames)), desc="Blending"):
        frame = frames[i].copy()
        

        pose_frame = frames[i].copy()
        if DO_ESTIMATE_POSE:
            pu, pv = anchors_pose[i]
            pu_i, pv_i = int(round(pu)), int(round(pv))
            if 0 <= pu_i < W0 and 0 <= pv_i < H0:
                cv2.circle(pose_frame, (pu_i, pv_i), 6, (0, 0, 255), -1)
                cv2.circle(pose_frame, (pu_i, pv_i), 10, (255, 255, 255), 2)
        vw_pose.write(pose_frame)
        
        
        flow_frame = frames[i].copy()
        fu, fv = anchors[i]                     
        fu_i, fv_i = int(round(fu)), int(round(fv))
        if 0 <= fu_i < W0 and 0 <= fv_i < H0:
            cv2.circle(flow_frame, (fu_i, fv_i), 6, (255, 0, 0), -1)       
            cv2.circle(flow_frame, (fu_i, fv_i), 10, (255, 255, 255), 2)   
        vw_flow.write(flow_frame)

        a = float(obj_angle_mod[i])    
        bgr, msk, pos, k0, k1, t = assets.sample_by_angle(a) 
        cx, cy = anchors[i]
        depth_i = depths[i]

        # source/mask
        if vsrc is not None and vmsk is not None:
            canvas_src = np.zeros_like(frame)
            canvas_msk = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # ROI
            h, w = bgr.shape[:2]
            x0 = int(np.clip(cx - w//2, 0, W0-1)); x1 = int(np.clip(x0 + w, 0, W0))
            y0 = int(np.clip(cy - h//2, 0, H0-1)); y1 = int(np.clip(y0 + h, 0, H0))
            bw, bh = (x1-x0), (y1-y0)
            if bw > 0 and bh > 0:
                canvas_src[y0:y1, x0:x1] = bgr[:bh, :bw]
                canvas_msk[y0:y1, x0:x1] = msk[:bh, :bw]
            vsrc.write(canvas_src)
            vmsk.write(cv2.cvtColor(canvas_msk, cv2.COLOR_GRAY2BGR))

        # blend
        if DRAW_POSE_DOT_ONLY:
            out = pose_frame
        else:
            out = overlay_center(frame, bgr, msk, (cx, cy), matcher=matcher)

        # out = overlay_center(frame, bgr, msk, (cx, cy), matcher=matcher)

        if DRAW_DEBUG_TEXT:
            txt = f"a={a:7.2f} pos={pos:7.3f} k0={k0:03d} k1={k1:03d} t={t:0.3f}"
            cv2.putText(out, txt, (24,48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30,220,30), 2, cv2.LINE_AA)

        vw.write(out)

    vw.release()
    vw_pose.release()
    vw_flow.release()

    print(f"pose dot video: {pose_path}")
    print(f"flow dot video: {flow_path}")


    if vsrc is not None:
        vsrc.release(); vmsk.release()
        print("debug video：", src_path, msk_path)

    print(f"output video：{out_path}")

if __name__ == "__main__":
    ensure_dir(OUT_ROOT)
    main()
