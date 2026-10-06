"""4차 과제: 독립적인 PPO 두 모델의 학습, 중단/재개, 비교 준비.

python train_assignment4.py --check
python train_assignment4.py --model a
python train_assignment4.py --model b
같은 명령을 다시 실행하면 해당 모델의 최신 run을 이어서 학습한다.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "assignment4"
PROFILES = {"a": {"lr": 5e-5}, "b": {"lr": 3e-4}}
REQUIRED_TIMESTEPS = 200_000
KST = timezone(timedelta(hours=9))
SOURCES = ("train.py", "train_assignment4.py", "algorithms/ppo.py",
           "utils/logger.py", "utils/buffer.py", "utils/networks.py",
           "utils/evaluator.py", "env/road_config.py", "env/mdp_config.py",
           "env/sumo_env.py", "env/road_builder.py")


def now():
    return datetime.now(KST).isoformat(timespec="seconds")


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def source_hashes():
    # Git의 Windows/Linux 줄바꿈 변환은 코드 변경으로 취급하지 않는다.
    return {name: hashlib.sha256((BASE / name).read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()
            for name in SOURCES}


def latest_run(model):
    candidates = list(RESULTS.glob(f"model_{model}_*/run_info.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.parent.name).parent


def check_environment():
    print(f"Python: {sys.executable}")
    print(f"Computer time (KST): {now()}")
    missing = [name for name in ("torch", "numpy", "gymnasium", "tensorboard",
                                "traci", "sumolib")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing)
                           + "\nRun: python -m pip install torch numpy gymnasium tensorboard traci sumolib")
    from torch.utils.tensorboard import SummaryWriter
    from env.sumo_env import SumoHighwayEnv
    from sumolib import checkBinary
    for name in ("sumo", "sumo-gui", "netconvert"):
        path = checkBinary(name)
        if not Path(path).is_file() and not shutil.which(path):
            raise RuntimeError(f"SUMO executable missing: {name}")
        print(f"{name}: {path}")
    print("OK: dependencies ready. No training started.")


def save_checkpoint(agent, run_dir, config, info):
    import numpy as np
    import torch
    np_state = np.random.get_state()
    state = {
        "policy": agent.policy.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "num_timesteps": agent.num_timesteps,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": [np_state[0], np_state[1].tolist(), np_state[2],
                      np_state[3], np_state[4]],
        "config": config,
    }
    tmp = run_dir / "checkpoint.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, run_dir / "checkpoint.pt")
    torch.save(agent.policy.state_dict(), run_dir / "model.pt.tmp")
    os.replace(run_dir / "model.pt.tmp", run_dir / "model.pt")
    info["trained_steps"] = agent.num_timesteps
    info["updated_at_kst"] = now()
    write_json(run_dir / "run_info.json", info)


def restore_checkpoint(agent, run_dir, config):
    import numpy as np
    import torch
    state = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if state["config"] != config:
        raise RuntimeError("Saved experiment configuration differs. Use --new-run for a new experiment.")
    agent.policy.load_state_dict(state["policy"])
    agent.optimizer.load_state_dict(state["optimizer"])
    agent.num_timesteps = state["num_timesteps"]
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    n = state["numpy_rng"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def train(args):
    from train import HPARAMS, REWARD, TOTAL_TIMESTEPS, EVAL_INTERVAL, EVAL_EPISODES
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env import road_builder
    from env.sumo_env import SumoHighwayEnv
    from algorithms.ppo import PPO
    from utils.logger import RunLogger

    if TOTAL_TIMESTEPS != REQUIRED_TIMESTEPS:
        raise RuntimeError("Assignment requires TOTAL_TIMESTEPS = 200000 in train.py. Do not change the training budget.")
    target = TOTAL_TIMESTEPS

    # JSON 정규화로 튜플/리스트의 저장 후 비교 차이를 제거한다.
    config = json.loads(json.dumps({
        "model": args.model, "hparams": {**HPARAMS, **PROFILES[args.model]},
        "reward": REWARD, "road": ROAD, "ego": EGO, "traffic": TRAFFIC,
        "simulation": SIMULATION, "observation": OBSERVATION, "action": ACTION,
        "eval_interval": EVAL_INTERVAL, "eval_episodes": EVAL_EPISODES,
    }))
    run_dir = None if args.new_run else latest_run(args.model)
    if run_dir:
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        if info["source_hashes"] != source_hashes():
            raise RuntimeError("Source files changed since the run began. Restore them or use --new-run.")
        if info["target_steps"] != target:
            raise RuntimeError("This run used a different training budget. Keep it and start a compliant run with --new-run.")
        if not (run_dir / "checkpoint.pt").is_file():
            raise RuntimeError("This run has no checkpoint. Start with --new-run.")
    else:
        stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S_%f")
        run_dir = RESULTS / f"model_{args.model}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        info = {"model": args.model, "started_at_kst": now(), "target_steps": target,
                "trained_steps": 0, "status": "prepared", "initialization": "random_from_scratch",
                "source_hashes": source_hashes(), "sessions": []}
        for name in SOURCES:
            dst = run_dir / "source" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BASE / name, dst)
        write_json(run_dir / "experiment_config.json", config)
        write_json(run_dir / "run_info.json", info)

    env = SumoHighwayEnv(cfg_path=str(run_dir / "sumo" / "highway.sumocfg"),
                         road=ROAD, ego=EGO, traffic=TRAFFIC, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, gui=False)
    agent = PPO(obs_dim=env.observation_space.shape[0],
                act_dim=env.action_space.shape[0], **config["hparams"])
    is_resuming = (run_dir / "checkpoint.pt").exists()
    if is_resuming:
        restore_checkpoint(agent, run_dir, config)
    if target <= agent.num_timesteps:
        info.update({"status": "complete", "trained_steps": agent.num_timesteps,
                     "target_steps": target})
        write_json(run_dir / "run_info.json", info)
        print(f"Already trained: {agent.num_timesteps} steps. Target: {target}.")
        print(f"Model: {run_dir / 'model.pt'}")
        return
    info["target_steps"] = target
    session = {"started_at_kst": now(), "start_step": agent.num_timesteps}
    info["sessions"].append(session)
    info["status"] = "running"
    logger = RunLogger(str(run_dir), resume_step=agent.num_timesteps if is_resuming else None)
    if logger.writer is None:
        logger.close()
        raise RuntimeError("TensorBoard logging is required for this assignment. Resolve the logging error before training.")
    logger.writer.add_text("experiment/settings", json.dumps(config, ensure_ascii=False, indent=2),
                           agent.num_timesteps)
    logger.writer.add_text("experiment/started_at_kst", info["started_at_kst"], agent.num_timesteps)
    save_checkpoint(agent, run_dir, config, info)
    print(f"Model {args.model.upper()}: lr={config['hparams']['lr']}, device={agent.device}")
    print(f"Run: {run_dir}")
    print(f"Start: {agent.num_timesteps}, target: {target} (rollout rounded: "
          f"{math.ceil(target / agent.n_steps) * agent.n_steps})")
    print(f"Session limit: {args.session_updates or 'all remaining'} updates. Ctrl+C saves progress.")
    completed_updates = 0
    started = time.monotonic()
    first_step = agent.num_timesteps
    interrupted = False

    def on_update(current):
        nonlocal completed_updates
        completed_updates += 1
        logger.save_csv()
        logger.writer.flush()
        save_checkpoint(current, run_dir, config, info)
        elapsed = time.monotonic() - started
        speed = (current.num_timesteps - first_step) / max(elapsed, 1e-9)
        remaining = max(0, math.ceil(target / current.n_steps) * current.n_steps - current.num_timesteps)
        print(f"    elapsed={elapsed / 60:.1f}min, {speed:.1f} steps/s, "
              f"estimated remaining={remaining / max(speed, 1e-9) / 60:.1f}min")
        return not args.session_updates or completed_updates < args.session_updates

    try:
        sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, str(run_dir / "sumo"))
        env.cfg_path = sumocfg
        agent.learn(env, total_timesteps=target, logger=logger, log_interval=1,
                    eval_interval=EVAL_INTERVAL, eval_episodes=EVAL_EPISODES,
                    on_update=on_update)
        info["status"] = "complete" if agent.num_timesteps >= target else "paused"
    except KeyboardInterrupt:
        interrupted = True
        info["status"] = "paused"
        print("\nInterrupted. Keeping the last fully completed update checkpoint.")
    except Exception:
        info["status"] = "error"
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            # rollout/optimizer의 도중에 중단되면 마지막 완료 업데이트로 되돌린다.
            restore_checkpoint(agent, run_dir, config)
            logger.train_rows = [r for r in logger.train_rows if int(r["steps"]) <= agent.num_timesteps]
            logger.eval_rows = [r for r in logger.eval_rows if int(r["steps"]) <= agent.num_timesteps]
            logger.close()
            if interrupted:
                # 미완료 rollout의 episode 이벤트를 즉시 TensorBoard에서 제외한다.
                cleaned_logger = RunLogger(str(run_dir), resume_step=agent.num_timesteps)
                cleaned_logger.close()
            session.update({"ended_at_kst": now(), "end_step": agent.num_timesteps,
                            "elapsed_seconds": round(time.monotonic() - started, 2)})
            save_checkpoint(agent, run_dir, config, info)
            print(f"Saved: {run_dir / 'model.pt'} ({agent.num_timesteps} trained steps)")
            print(f"Resume: python train_assignment4.py --model {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Assignment 4: train two fresh PPO models in resumable sessions.")
    parser.add_argument("action", nargs="?",
                        choices=["check", "a", "b", "status", "tensorboard", "drive-a", "drive-b"])
    parser.add_argument("--check", action="store_true", help="Check dependencies; no training")
    parser.add_argument("--model", choices=PROFILES)
    parser.add_argument("--session-updates", type=int, default=10,
                        help="Updates per invocation (default 10); 0 = run to target")
    parser.add_argument("--new-run", action="store_true", help="Start another independent run")
    parser.add_argument("--status", action="store_true", help="Show assignment run progress")
    parser.add_argument("--tensorboard", action="store_true", help="Show only assignment runs")
    parser.add_argument("--drive", choices=PROFILES, help="Replay the latest assignment model")
    args = parser.parse_args()
    if args.action in PROFILES:
        args.model = args.action
    elif args.action in ("check", "status", "tensorboard"):
        setattr(args, args.action, True)
    elif args.action and args.action.startswith("drive-"):
        args.drive = args.action[-1]
    if args.session_updates < 0:
        parser.error("--session-updates must be >= 0")
    if args.check:
        check_environment()
    elif args.status:
        for model in PROFILES:
            run_dir = latest_run(model)
            if run_dir is None:
                print(f"Model {model.upper()}: not started")
                continue
            info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
            print(f"Model {model.upper()}: {info['status']}, "
                  f"{info['trained_steps']}/{info['target_steps']} steps")
            print(f"  started: {info['started_at_kst']}\n  run: {run_dir}")
    elif args.tensorboard:
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir", str(RESULTS),
                        "--host", "127.0.0.1", "--port", "6006", "--reload_interval", "2"],
                       check=True)
    elif args.drive:
        run_dir = latest_run(args.drive)
        if run_dir is None or not (run_dir / "model.pt").is_file():
            parser.error("No model yet. Train this assignment model first.")
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        if info["trained_steps"] == 0:
            parser.error("The model has no completed training updates yet.")
        subprocess.run([sys.executable, str(BASE / "test.py"), str(run_dir / "model.pt"),
                        "--algorithm", "ppo", "--episodes", "1"], cwd=BASE, check=True)
    elif args.model:
        train(args)
    else:
        parser.error("Choose --check, --status, --tensorboard, --drive a/b or --model a/b")


if __name__ == "__main__":
    main()
