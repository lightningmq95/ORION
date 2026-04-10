#!/usr/bin/env python3
"""
eval.py — Odometry recording, evaluation, and visualization.

Records wheel odometry and EKF-fused pose, with ground truth for evaluation.
Supports infinite recording with live plotting and automatic plot saving.

Workflow:
  1. Record: python eval.py --record --output ./runs
     → Records indefinitely with live plot, saves CSVs and plots
  2. Reload: python eval.py --load ./runs/10_April_2026_14_30_45 --visualize
     → Loads CSVs and displays saved plots with RMSE analysis

Usage:
  python eval.py --record --output ./runs
  python eval.py --load ./runs/10_April_2026_14_30_45 --visualize
"""

import os, sys, csv, signal, atexit, argparse, threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

# ─────────────────────────── constants ───────────────────────────────────────

# name → (color, linestyle)
STYLES: Dict[str, tuple] = {
    'Wheel Odometry': ('green', '-'),
    'EKF Fused':      ('black', '-'),
    'Ground Truth':   ('purple', '-'),
}

# individual subplot grid positions — show raw native-frame data
SUBPLOTS = [
    ('Wheel Odometry', 0, 0),
    ('EKF Fused',      0, 1),
    ('Ground Truth',   1, 0),
]

# ─────────────────────────── data model ──────────────────────────────────────

@dataclass
class SensorData:
    name:      str
    color:     str = 'gray'
    linestyle: str = '-'
    _pts:      List = field(default_factory=list)
    _lock:     threading.Lock = field(default_factory=threading.Lock,
                                      repr=False, compare=False)

    def add(self, ts: float, x: float, y: float):
        with self._lock: self._pts.append((ts, x, y))

    def xy(self):
        with self._lock:
            if not self._pts: return np.array([]), np.array([])
            a = np.array(self._pts); return a[:, 1], a[:, 2]

    def ts(self):
        with self._lock:
            if not self._pts: return np.array([])
            return np.array(self._pts)[:, 0]

    def count(self): return len(self._pts)

    def gaps(self, thr=2.0):
        t = sorted(self.ts())
        return [(t[i-1], t[i], t[i]-t[i-1]) for i in range(1, len(t)) if t[i]-t[i-1] > thr]

    @property
    def points(self): return self._pts   # CSV compat


def _mk(name): return SensorData(name, *STYLES.get(name, ('gray', '-')))
def _dist(x, y): return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))) if len(x) > 1 else 0.0

def _rmse(sx, sy, st, rx, ry, rt, max_dt=0.15):
    if len(sx) < 5 or len(rx) < 5: return None
    # Origin-align both trajectories to their first point so that a different
    # coordinate origin (e.g. Gazebo world frame vs EKF map frame) does not
    # inflate the error.  RMSE then measures drift/shape, not absolute offset.
    sx = sx - sx[0]; sy = sy - sy[0]
    rx = rx - rx[0]; ry = ry - ry[0]
    pairs = []
    for i, ts in enumerate(st):
        j = int(np.argmin(np.abs(rt - ts)))
        if abs(rt[j] - ts) < max_dt: pairs.append((sx[i]-rx[j], sy[i]-ry[j]))
    if len(pairs) < 3: return None
    ex, ey = np.array(pairs).T
    return float(np.sqrt(np.mean(ex**2 + ey**2)))

# ─────────────────────────── ROS2 recorder ───────────────────────────────────

class Recorder(Node):
    """
    Records odometry from three sources:
      - /odom: Wheel odometry
      - /odom_fused: EKF-fused pose
      - /ground_truth: Ground truth pose from Gazebo
    """

    def __init__(self):
        super().__init__('odom_visualizer')
        self.recording  = True

        # Dict for all odometry data
        self.data: Dict[str, SensorData] = {
            'Wheel Odometry': _mk('Wheel Odometry'),
            'EKF Fused':      _mk('EKF Fused'),
            'Ground Truth':   _mk('Ground Truth'),
        }

        def pose_cb(sd):
            def cb(msg: PoseStamped):
                if self.recording:
                    sd.add(self._ts(msg.header.stamp), msg.pose.position.x, msg.pose.position.y)
            return cb

        # Subscriptions
        self.create_subscription(Odometry,                  '/odom',                     self._cb_odom,      10)
        self.create_subscription(Odometry,                  '/odom_fused',               self._cb_ekf,       10)
        self.create_subscription(PoseStamped,               '/ground_truth',             pose_cb(self.data['Ground Truth']), 10)

    @staticmethod
    def _ts(s): return s.sec + s.nanosec * 1e-9

    def _cb_ekf(self, msg: Odometry):
        if self.recording:
            ts = self._ts(msg.header.stamp)
            x  = msg.pose.pose.position.x
            y  = msg.pose.pose.position.y
            self.data['EKF Fused'].add(ts, x, y)

    def _cb_odom(self, msg: Odometry):
        if self.recording:
            ts = self._ts(msg.header.stamp)
            x  = msg.pose.pose.position.x
            y  = msg.pose.pose.position.y
            self.data['Wheel Odometry'].add(ts, x, y)

    def stop(self): self.recording = False

# ─────────────────────────── evaluator ───────────────────────────────────────

class Evaluator:
    """Holds sensor data. Provides RMSE, stats, CSV, and plots."""

    def __init__(self):
        self.data:       Dict[str, SensorData] = {}
        self.rmse_vals:  dict = {}
        self._tag = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── RMSE ─────────────────────────────────────────────────────────────────

    def compute_rmse(self):
        gt = self.data.get('Ground Truth')
        if gt is None or not gt.count(): print("No Ground Truth — skip RMSE."); return
        rx, ry, rt = *gt.xy(), gt.ts()
        for name, sd in self.data.items():
            if name == 'Ground Truth' or not sd.count(): continue
            sx, sy, st = *sd.xy(), sd.ts()
            offset = np.median(rt) - np.median(st)
            if abs(offset) > 1000.0: st = st + offset
            v = _rmse(sx, sy, st, rx, ry, rt)
            if v is not None:
                self.rmse_vals[name] = v
                print(f"  RMSE {name:<25s}: {v:.4f} m")

    # ── stats ─────────────────────────────────────────────────────────────────

    def print_stats(self):
        print("\n" + "="*55 + "\nODOMETRY EVALUATION\n" + "="*55)
        if self.rmse_vals:
            print("\nRMSE vs Ground Truth (origin-aligned):")
            for n, v in sorted(self.rmse_vals.items(), key=lambda x: x[1]):
                print(f"  {n:<25s}: {v:.4f} m")
        print("\nSensor summary:")
        for name, sd in self.data.items():
            if not sd.count(): continue
            x, y = sd.xy(); t = sd.ts()
            dur = t[-1]-t[0] if len(t) > 1 else 0; n = sd.count()
            g = sd.gaps(); avail = 100*(1-sum(d for _,_,d in g)/dur) if g and dur else 100
            print(f"  {name:<25s}: {n} pts  {dur:.1f}s  "
                  f"{n/max(dur,1e-9):.1f}Hz  {_dist(x,y):.1f}m  {avail:.0f}% avail")

    # ── CSV ───────────────────────────────────────────────────────────────────

    def save_csv(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        for name, sd in self.data.items():
            slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            fp   = os.path.join(out_dir, f'{slug}.csv')
            with open(fp, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['timestamp', 'x', 'y'])
                w.writerows(sd.points)
            print(f"  Saved {name} → {os.path.basename(fp)}")

    def load_csv(self, in_dir: str):
        FILE_MAP = {
            'wheel_odometry': 'Wheel Odometry',
            'ekf_fused':      'EKF Fused',
            'ground_truth':   'Ground Truth',
        }
        for base, name in FILE_MAP.items():
            fp = os.path.join(in_dir, f'{base}.csv')
            if not os.path.exists(fp): continue
            sd = _mk(name)
            with open(fp) as f:
                for row in csv.DictReader(f):
                    sd.add(float(row['timestamp']), float(row['x']), float(row['y']))
            self.data[name] = sd
            print(f"  Loaded {name} — {sd.count()} pts")

    # ── plot helpers ──────────────────────────────────────────────────────────

    def _draw(self, ax, sd: SensorData, lw=1.5, alpha=0.8):
        x, y = sd.xy()
        if not len(x): return
        ax.plot(x, y, color=sd.color, ls=sd.linestyle, lw=lw, alpha=alpha, label=sd.name)
        ax.scatter([x[0]], [y[0]], color=sd.color, marker='o', s=70, zorder=5, alpha=alpha)
        ax.scatter([x[-1]], [y[-1]], color=sd.color, marker='s', s=70, zorder=5, alpha=alpha)

    def _subplot_raw(self, ax, name: str, gap_thr=1.0):
        """Individual subplot — sensor trajectory."""
        sd = self.data.get(name)
        c  = STYLES.get(name, ('gray','-'))[0]
        if sd is None or not sd.count():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(f'{name}\n[no data]', fontsize=9, color='gray')
            ax.grid(True, alpha=0.3); return
        x, y = sd.xy(); t = sd.ts()
        dur  = t[-1]-t[0] if len(t) > 1 else 0
        g    = sd.gaps(gap_thr); avail = 100*(1-sum(d for _,_,d in g)/dur) if g and dur else 100
        ax.plot(x, y, color=c, lw=2, alpha=0.85)
        ax.scatter([x[0]], [y[0]], color=c, marker='o', s=80, zorder=5)
        ax.scatter([x[-1]], [y[-1]], color=c, marker='s', s=80, zorder=5)
        for gs, ge, _ in g:
            ib = np.argmin(np.abs(t-gs)); ia = np.argmin(np.abs(t-ge))
            ax.plot([x[ib], x[ia]], [y[ib], y[ia]], 'r--', lw=1.5, alpha=0.7)
        info = (f'{sd.count()} pts | {sd.count()/max(dur,1e-9):.1f} Hz\n'
                f'{_dist(x,y):.1f} m | {avail:.0f}% avail') if dur else f'{sd.count()} pts'
        ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'{name}', fontsize=9, color=c, fontweight='bold')
        ax.grid(True, alpha=0.3); ax.axis('equal')
        ax.set(xlabel='X (m)', ylabel='Y (m)'); ax.tick_params(labelsize=8)

    def _subplot_enu(self, ax):
        """Combined plot — all sensors."""
        priority = ['Ground Truth', 'Wheel Odometry', 'EKF Fused']
        for name in priority:
            sd = self.data.get(name)
            if sd is None or not sd.count(): continue
            lw = 2.5 if name == 'EKF Fused' else 1.5
            al = 0.9  if name == 'EKF Fused' else 0.7
            self._draw(ax, sd, lw=lw, alpha=al)
        ax.set(xlabel='X (m)', ylabel='Y (m)',
               title='All sensors\n○ start  □ end')
        ax.grid(True, alpha=0.3); ax.axis('equal'); ax.legend(loc='upper right', fontsize=9)
        if self.rmse_vals:
            txt = 'RMSE vs GT (origin-aligned)\n' + '\n'.join(
                f'  {n}: {v:.4f} m' for n, v in sorted(self.rmse_vals.items(), key=lambda x: x[1]))
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8, va='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ── public plots ──────────────────────────────────────────────────────────

    def plot(self, gap_thr=1.0):
        """
        2×2 grid of individual subplots plus combined plot.
        """
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Odometry Evaluation',
                     fontsize=13, fontweight='bold')
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        for i, (name, row, col) in enumerate(SUBPLOTS):
            self._subplot_raw(fig.add_subplot(gs[row, col]), name, gap_thr)
        self._subplot_enu(fig.add_subplot(gs[1, 1]))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def plot_availability(self, gap_thr=1.0):
        """Horizontal bar chart of sensor availability over time."""
        fig, ax = plt.subplots(figsize=(12, 4))
        names   = list(self.data.keys())
        ts_all  = [sd.ts() for sd in self.data.values() if sd.count()]
        t0 = min(t.min() for t in ts_all) if ts_all else 0
        t1 = max(t.max() for t in ts_all) if ts_all else 1
        for i, (name, sd) in enumerate(self.data.items()):
            c = STYLES.get(name, ('gray',))[0]
            if not sd.count():
                ax.barh(i, t1-t0, left=t0, height=0.6, color='lightgray', alpha=0.4); continue
            t = sorted(sd.ts()); seg = t[0]
            for j in range(1, len(t)):
                if t[j]-t[j-1] > gap_thr:
                    ax.barh(i, t[j-1]-seg, left=seg, height=0.6, color=c, alpha=0.8)
                    ax.barh(i, t[j]-t[j-1], left=t[j-1], height=0.6,
                            color='lightcoral', alpha=0.4, hatch='///')
                    seg = t[j]
            ax.barh(i, t[-1]-seg, left=seg, height=0.6, color=c, alpha=0.8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set(xlabel='Time (s)', title='Sensor availability (hatched = gap)')
        ax.grid(True, axis='x', alpha=0.3); ax.set_xlim(t0, t1)
        plt.tight_layout(); return fig

# ─────────────────────────── live plot ───────────────────────────────────────

class LivePlot:
    """Real-time figure with live sensor trajectories."""

    def __init__(self, data: Dict[str, SensorData], interval_ms=600):
        self.data = data
        self._rec = True
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

        # individual subplots
        self.ax_ind = {}
        for name, row, col in SUBPLOTS:
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_title(f'{name}', fontsize=9, color=STYLES[name][0], fontweight='bold')
            ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'datalim')
            ax.set(xlabel='X (m)', ylabel='Y (m)'); ax.tick_params(labelsize=8)
            self.ax_ind[name] = ax

        # combined plot (bottom right)
        self.ax_traj = self.fig.add_subplot(gs[1, 1])
        self.ax_traj.set(title='Sensor Trajectories',
                         xlabel='X (m)', ylabel='Y (m)')
        self.ax_traj.grid(True, alpha=0.3)
        self._lines, self._ds, self._de = {}, {}, {}
        for name, (c, ls) in STYLES.items():
            lw = 2.5 if name == 'EKF Fused' else 1.5
            al = 0.95 if name == 'EKF Fused' else 0.7
            self._lines[name], = self.ax_traj.plot([], [], color=c, ls=ls, lw=lw, alpha=al, label=name)
            self._ds[name], = self.ax_traj.plot([], [], 'o', color=c, ms=7, zorder=5)
            self._de[name], = self.ax_traj.plot([], [], 's', color=c, ms=7, zorder=5)
        self.ax_traj.legend(loc='upper left', fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._anim = animation.FuncAnimation(self.fig, self._update,
                                             interval=interval_ms, blit=False,
                                             cache_frame_data=False)

    def _update(self, _):
        # combined — all sensors
        for name, line in self._lines.items():
            sd = self.data.get(name)
            x, y = (sd.xy() if sd else (np.array([]), np.array([])))
            if len(x):
                line.set_data(x, y)
                self._ds[name].set_data([x[0]], [y[0]])
                self._de[name].set_data([x[-1]], [y[-1]])
            else:
                line.set_data([], [])
                self._ds[name].set_data([], [])
                self._de[name].set_data([], [])
        self.ax_traj.relim(); self.ax_traj.autoscale_view()

        # individual subplots
        for name, ax in self.ax_ind.items():
            sd = self.data.get(name)
            if sd is None or not sd.count(): continue
            x, y = sd.xy(); c, ls = STYLES[name]
            ax.clear()
            ax.plot(x, y, color=c, ls=ls, lw=2, alpha=0.9)
            ax.scatter([x[0]], [y[0]], color=c, marker='o', s=70, zorder=5)
            ax.scatter([x[-1]], [y[-1]], color=c, marker='s', s=70, zorder=5)
            ax.text(0.03, 0.97, f'pts:{len(x)}  dist:{_dist(x,y):.1f}m',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_title(f'{name}', fontsize=9, color=c, fontweight='bold')
            ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'datalim')
            ax.set(xlabel='X (m)', ylabel='Y (m)'); ax.tick_params(labelsize=8)

        st  = '[● REC]' if self._rec else '[■ DONE]'
        self.fig.suptitle(f'Live Odometry  {st}',
                          fontsize=12, fontweight='bold', color='red' if self._rec else 'black')

    def stop(self): self._rec = False
    def show(self): plt.show()

# ─────────────────────────── record session ──────────────────────────────────

def record(output_dir=None, live=True) -> 'Evaluator':
    rclpy.init()
    rec = Recorder()
    threading.Thread(target=rclpy.spin, args=(rec,), daemon=True).start()
    print("Recording — press Ctrl+C to stop.")

    def _cleanup():
        rec.stop()
        try:
            if rec.context.ok(): rec.destroy_node()
        except Exception: pass
        try:
            if rclpy.ok(): rclpy.shutdown()
        except Exception: pass

    atexit.register(_cleanup)

    def _build() -> Evaluator:
        rec.stop()
        ev = Evaluator()
        ev.data = rec.data
        ev.compute_rmse()
        if output_dir:
            ev.save_csv(output_dir)
            # Auto-generate and save plots
            print("\nGenerating plots...")
            fig = ev.plot()
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            fig.savefig(os.path.join(plots_dir, 'trajectories.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved → {plots_dir}/trajectories.png")
        return ev

    stop_evt = threading.Event(); interrupted = threading.Event()

    def _sig(s, f):
        interrupted.set(); stop_evt.set(); rec.stop()
        if live: plt.close('all')

    signal.signal(signal.SIGINT, _sig)

    if live:
        lp = LivePlot(rec.data)
        try: plt.show(block=True)
        except KeyboardInterrupt: pass
        finally: stop_evt.set(); rec.stop(); lp.stop(); plt.close('all')
    else:
        try: stop_evt.wait()
        except KeyboardInterrupt: pass
        finally: stop_evt.set(); rec.stop()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return _build()

# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    clean_argv = remove_ros_args(sys.argv)[1:] 

    ap = argparse.ArgumentParser(
        description='Odometry evaluation & visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n"
                "  python eval.py --record --output ./runs\n"
                "  python eval.py --load ./runs/14_March_2026_10_30_00 --visualize\n"))
    
    ap.add_argument('--record',        action='store_true')
    ap.add_argument('--output',        type=str,   default=None)
    ap.add_argument('--load',          type=str,   default=None)
    ap.add_argument('--visualize',     action='store_true')
    ap.add_argument('--no-live',       action='store_true')
    ap.add_argument('--gap-threshold', type=float, default=1.0)
    ap.add_argument('--save-plots',    type=str,   default=None)
    
    # 2. Parse the CLEANED arguments list
    args = ap.parse_args(clean_argv)

    if args.record and not args.output:
        print("ERROR: --output required with --record")
        return

    data_dir = None

    if args.record:
        stamp    = datetime.now().strftime('%d_%B_%Y_%H_%M_%S')
        data_dir = os.path.join(args.output, stamp)
        print(f"Saving to: {data_dir}")
        # record() handles its own rclpy.init()/shutdown logic internally
        ev = record(output_dir=data_dir, live=not args.no_live)
        print(f"\n✓ Recording complete and saved to: {data_dir}")
        print(f"\nTo view plots again, run:")
        print(f"  python eval.py --load {data_dir} --visualize\n")
    elif args.load:
        ev = Evaluator()
        data_dir = args.load
        ev.load_csv(args.load)
        ev.compute_rmse()
    else:
        print("Nothing to do — use --record or --load")
        return

    ev.print_stats()

    if args.visualize or args.record:
        fig = ev.plot(gap_thr=args.gap_threshold)
        if args.record:
            print("\nDisplaying trajectories (close window to exit)...")
        
        def _sig(s, f): 
            plt.close('all')
            sys.exit(0)
            
        signal.signal(signal.SIGINT, _sig)
        
        try: 
            plt.show(block=True)
        except KeyboardInterrupt: 
            pass
        finally: 
            plt.close('all')
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)