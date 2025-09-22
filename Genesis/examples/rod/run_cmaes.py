import os
import json
import csv
import time
import pickle
from typing import Tuple, List, Optional, Sequence
from argparse import ArgumentParser



import numpy as np
import cma

from train_env_wiring_ring import Train_Env_Wiring_ring  # keep if you still want the example main
from train_env_wiring_post import Train_Env_Wiring_post
from train_env_lifting import Train_Env_Lifting
from train_env_slingshot import Train_Env_Slingshot
from train_env_wireart import Train_Env_Wireart

from train_env_coiling import Train_Env_Coiling
from train_env_gathering import Train_Env_Gathering
from train_env_separation import Train_Env_Separation
from train_env_wrapping import Train_Env_Wrapping


# ----------------------------
# Helpers: shape & constraints
# ----------------------------
def reshape_to_traj(x: np.ndarray, n_steps: int, act_dim: int) -> np.ndarray:
    return x.reshape(n_steps, act_dim)

def _as_per_comp_array(per_comp_bound: Optional[Sequence[float]], act_dim: int) -> np.ndarray:
    if per_comp_bound is None:
        return np.full((act_dim,), np.inf, dtype=np.float32)
    if np.isscalar(per_comp_bound):
        return np.full((act_dim,), float(per_comp_bound), dtype=np.float32)
    arr = np.asarray(per_comp_bound, dtype=np.float32).reshape(-1)
    if arr.size != act_dim:
        raise ValueError(f"per_comp_bound length {arr.size} != act_dim {act_dim}")
    return arr

def project_deltas(traj: np.ndarray,
                   per_comp_bound: Optional[Sequence[float]],
                   max_l2_per_step: Optional[float]) -> np.ndarray:
    n_steps, act_dim = traj.shape
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    if np.isfinite(pcb).any():
        traj = np.clip(traj, -pcb, pcb)
    if max_l2_per_step is not None and np.isfinite(max_l2_per_step):
        norms = np.linalg.norm(traj, axis=1, keepdims=True)
        scale = np.ones_like(norms, dtype=traj.dtype)
        over = norms > max_l2_per_step
        scale[over] = max_l2_per_step / (norms[over] + 1e-12)
        traj = traj * scale
    return traj


# ----------------------------
# Parallel evaluation (batch)
# ----------------------------
def evaluate_batch(env, traj_list: List[np.ndarray]) -> np.ndarray:
    n_envs = env.n_envs
    n_steps = traj_list[0].shape[0]
    act_dim = traj_list[0].shape[1]
    trajs = np.zeros((n_envs, n_steps, act_dim), dtype=np.float32)
    for i, tr in enumerate(traj_list):
        trajs[i] = tr
    rewards = env.eval_traj(trajs)
    return np.asarray(rewards, dtype=np.float32)


# ----------------------------
# Logging helpers
# ----------------------------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _maybe_write_header(path: str, header: List[str]):
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if needs_header:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def _append_rewards(log_dir: str, iteration: int, rewards: np.ndarray):
    path = os.path.join(log_dir, "rewards_all.csv")
    _maybe_write_header(path, ["iter", "idx", "reward"])
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for idx, r in enumerate(rewards.tolist()):
            w.writerow([iteration, idx, float(r)])

def _append_summary(log_dir: str, iteration: int, pop: int, n_chunks: int,
                    mean: float, std: float, rmin: float, rmax: float,
                    best_so_far: float, sigma_now: float,
                    t_iter_sec: float, t_total_sec: float):
    path = os.path.join(log_dir, "summary.csv")
    _maybe_write_header(path, [
        "iter", "pop", "chunks", "mean", "std", "min", "max",
        "best_so_far", "sigma", "t_iter_s", "t_total_s"
    ])
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            iteration, pop, n_chunks, mean, std, rmin, rmax,
            best_so_far, sigma_now, t_iter_sec, t_total_sec
        ])

def _save_best_traj(log_dir: str, best_traj: np.ndarray):
    np.save(os.path.join(log_dir, "best_traj.npy"), best_traj)

def _save_run_config(log_dir: str, cfg: dict):
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ----------------------------
# CMA-ES checkpoint helpers
# ----------------------------
def _ckpt_dir(work_dir: Optional[str], trial_name: Optional[str]) -> Optional[str]:
    if work_dir is None or trial_name is None:
        return None
    return os.path.join(work_dir, trial_name)

def _ckpt_paths(work_dir: Optional[str], trial_name: Optional[str]):
    d = _ckpt_dir(work_dir, trial_name)
    if d is None:
        return None, None
    _ensure_dir(d)
    return os.path.join(d, "cmaes_ckpt.pkl"), os.path.join(d, "resume_meta.json")

def _save_cma_ckpt(es, work_dir: Optional[str], trial_name: Optional[str], iter_idx: int):
    pkl_path, meta_path = _ckpt_paths(work_dir, trial_name)
    if pkl_path is None:
        return
    # Save CMA-ES opaque state
    with open(pkl_path, "wb") as f:
        f.write(es.pickle_dumps())
    # Save minimal meta so logs can continue with correct iteration index
    with open(meta_path, "w") as f:
        json.dump({"iter": iter_idx}, f)

def _load_cma_ckpt(work_dir: Optional[str], trial_name: Optional[str]):
    pkl_path, meta_path = _ckpt_paths(work_dir, trial_name)
    if pkl_path is None:
        return None, 0
    if not os.path.exists(pkl_path):
        return None, 0
    with open(pkl_path, "rb") as f:
        es = pickle.load(f)
    start_iter = 0
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                start_iter = int(json.load(f).get("iter", 0)) + 1
        except Exception:
            start_iter = 0
    return es, start_iter


# ----------------------------
# CMA-ES optimization (general)
# ----------------------------
def _infer_act_dim(env) -> Optional[int]:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "act_dim"):
        return int(env.act_dim)
    if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
        return int(env.action_space.shape[0])
    return None

def _infer_n_steps(env) -> Optional[int]:
    for attr in ("n_steps", "traj_len", "horizon", "T"):
        if hasattr(env, attr):
            return int(getattr(env, attr))
    return None


def optimize_wiring_trajectory(
    env,
    n_steps: Optional[int] = None,
    act_dim: Optional[int] = None,
    popsize: Optional[int] = None,
    sigma0: float = 0.01,
    per_comp_bound: Optional[Sequence[float]] = 0.01,
    l2_bound: Optional[float] = None,
    max_iters: int = 200,
    seed: int = 42,
    log_dir: Optional[str] = None,
    # NEW: checkpointing
    work_dir: Optional[str] = None,
    trial_name: Optional[str] = None,
    resume: bool = False,
    save_every: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    Adds CMA-ES checkpointing via (work_dir/trial_name)/cmaes_ckpt.pkl.
    - If resume=True and the file exists, loads the CMA state and continues.
    - Saves checkpoint every `save_every` iters and at the end.

    Other behavior: general shapes, logging, and optional bound inference unchanged.
    """
    # Resolve shapes
    if act_dim is None:
        act_dim = _infer_act_dim(env)
    if n_steps is None:
        n_steps = _infer_n_steps(env)
    if act_dim is None:
        raise ValueError("act_dim could not be inferred; please pass act_dim explicitly.")
    if n_steps is None:
        raise ValueError("n_steps could not be inferred; please pass n_steps explicitly.")

    # Resolve l2 bound (optional)
    if l2_bound is None and hasattr(env, "l2_bound"):
        l2_bound = float(getattr(env, "l2_bound"))

    # Resolve log_dir
    if log_dir is None and hasattr(env, "log_dir"):
        log_dir = getattr(env, "log_dir")
    if log_dir is not None:
        _ensure_dir(log_dir)
        _save_run_config(log_dir, {
            "n_steps": n_steps,
            "act_dim": act_dim,
            "popsize": popsize,
            "sigma0": sigma0,
            "per_comp_bound": (float(per_comp_bound) if np.isscalar(per_comp_bound)
                               else (list(per_comp_bound) if per_comp_bound is not None else None)),
            "l2_bound": l2_bound,
            "max_iters": max_iters,
            "seed": seed,
            "work_dir": work_dir,
            "trial_name": trial_name,
        })

    dim = n_steps * act_dim
    pcb = _as_per_comp_array(per_comp_bound, act_dim)
    lower, upper = [], []
    for _ in range(n_steps):
        lower.extend((-pcb).tolist())
        upper.extend((+pcb).tolist())

    # Try to resume CMA-ES
    es = None
    start_iter = 0
    if resume:
        es_loaded, start_iter = _load_cma_ckpt(work_dir, trial_name)
        if es_loaded is not None:
            es = es_loaded
            # quick sanity: check dimension matches
            if getattr(es, "N", dim) != dim:
                raise ValueError(f"Loaded CMA-ES dimension {getattr(es, 'N', None)} "
                                 f"does not match expected dim {dim}.")
            # Note: popsize is internal in es.opts; we trust the checkpoint.
            print(f"[resume] Loaded CMA-ES from iteration {start_iter} "
                  f"with dim={dim}, expected max_iters={max_iters}.")
        else:
            print("[resume] No checkpoint found; starting fresh.")

    # Fresh CMA-ES if not resumed
    if es is None:
        es = cma.CMAEvolutionStrategy(
            x0=[0.0] * dim,
            sigma0=sigma0,
            inopts={
                'bounds': [lower, upper],
                'popsize': popsize,
                'seed': seed,
                'CMA_elitist': True,
                'verb_disp': 0,
            }
        )

    best_traj = None
    best_reward = -np.inf

    batch_size = env.n_envs
    it = start_iter
    t0_all = time.time()

    # If resuming, keep the previous total time in resume_meta (optional)
    print(f"{'iter':>5} | {'pop':>4} | {'chunks':>6} | {'mean':>8} | {'std':>8} | "
          f"{'min':>8} | {'max':>8} | {'best':>8} | {'sigma':>7} | {'t_iter(s)':>8} | {'t_total(s)':>9}")

    while not es.stop() and it < max_iters:
        t_iter = time.time()
        X = es.ask()
        pop = len(X)
        n_chunks = (pop + batch_size - 1) // batch_size

        all_rewards = []
        for ci, start in enumerate(range(0, pop, batch_size), 1):
            t_chunk = time.time()
            chunk = X[start:start + batch_size]
            trajs = []
            for x in chunk:
                x_arr = np.asarray(x, dtype=np.float32)
                tr = reshape_to_traj(x_arr, n_steps, act_dim)
                tr = project_deltas(tr, per_comp_bound, l2_bound)
                trajs.append(tr)
            rewards = evaluate_batch(env, trajs)
            all_rewards.extend(rewards.tolist())
            print(f"  └─ chunk {ci:>2}/{n_chunks}: {len(chunk):>3} evals | t={time.time() - t_chunk:.3f}s")

        all_rewards = np.asarray(all_rewards, dtype=np.float32)

        # Log raw rewards for this generation
        if log_dir is not None:
            _append_rewards(log_dir, it, all_rewards)

        # CMA-ES minimizes; negate to maximize reward
        es.tell(X, (-all_rewards).tolist())

        # Track best of gen
        gen_best_idx = int(np.argmax(all_rewards))
        gen_best_reward = float(all_rewards[gen_best_idx])
        gen_best_x = np.asarray(X[gen_best_idx], dtype=np.float32)
        gen_best_traj = project_deltas(
            reshape_to_traj(gen_best_x, n_steps, act_dim),
            per_comp_bound, l2_bound
        )

        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_traj = gen_best_traj.copy()
            if log_dir is not None:
                _save_best_traj(log_dir, best_traj)

        # Iteration summary
        m = float(all_rewards.mean()) if all_rewards.size else float('nan')
        s = float(all_rewards.std()) if all_rewards.size else float('nan')
        mn = float(all_rewards.min()) if all_rewards.size else float('nan')
        mx = float(all_rewards.max()) if all_rewards.size else float('nan')
        try:
            sigma_now = float(es.sigma)
        except Exception:
            sigma_now = float(es.sigma0) if hasattr(es, 'sigma0') else float('nan')

        t_iter_sec = time.time() - t_iter
        t_total_sec = time.time() - t0_all

        print(f"{it:5d} | {pop:4d} | {n_chunks:6d} | {m:8.4f} | {s:8.4f} | "
              f"{mn:8.4f} | {mx:8.4f} | {best_reward:8.4f} | {sigma_now:7.4f} | "
              f"{t_iter_sec:8.3f} | {t_total_sec:9.3f}")

        if log_dir is not None:
            _append_summary(log_dir, it, pop, n_chunks, m, s, mn, mx, best_reward, sigma_now, t_iter_sec, t_total_sec)

        # Save checkpoint periodically
        if save_every > 0 and (it % save_every == 0):
            _save_cma_ckpt(es, work_dir, trial_name, it)

        it += 1

    # Final checkpoint
    _save_cma_ckpt(es, work_dir, trial_name, it - 1)

    return best_traj, best_reward


# ----------------------------
# Example usage
# ----------------------------

def _build_env(task: str, log_dir: str, n_envs: int):
    task = task.lower()
    task_to_env = {
        "wiring":    Train_Env_Wiring_ring,
        "wiring_post": Train_Env_Wiring_post,
        "lifting":   Train_Env_Lifting,
        "slingshot": Train_Env_Slingshot,
        "wireart":   Train_Env_Wireart,

        "coiling":   Train_Env_Coiling,
        "gathering": Train_Env_Gathering,
        "separation": Train_Env_Separation,
        "wrapping":  Train_Env_Wrapping,
    }
    if task not in task_to_env:
        raise ValueError(f"Unknown task '{task}'. Valid: {sorted(task_to_env.keys())}")
    EnvCls = task_to_env[task]
    # Most envs accept (task=..., log_dir=..., n_envs=...), but if yours differ, tweak here.
    try:
        return EnvCls(task=task, log_dir=log_dir, n_envs=n_envs)
    except TypeError:
        # Fallback if the env ctor only takes (log_dir, n_envs) or similar
        try:
            return EnvCls(log_dir=log_dir, n_envs=n_envs)
        except TypeError:
            return EnvCls()
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="wiring",
                   help="Task / environment to optimize.")
    args = parser.parse_args()
    trial_name = f"trial_{args.task}"
    env = _build_env(args.task, f"logs/{args.task}", 10)

    n_steps = 10

    best_traj, best_reward = optimize_wiring_trajectory(
        env,
        n_steps=n_steps,
        act_dim=None,           # infer if available
        popsize=100,
        sigma0=0.005,
        per_comp_bound=0.02,
        l2_bound=0.03,          # use env.l2_bound if present
        max_iters=15,
        seed=123,
        log_dir=f"logs/{args.task}",
        # NEW: checkpoint controls
        work_dir="checkpoints",
        trial_name=trial_name,
        resume=True,            # set True to load if checkpoint exists
        save_every=1,           # save each generation
    )
