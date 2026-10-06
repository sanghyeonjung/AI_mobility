"""학습하지 않고 분할 실행/복원/로그/모델 선택 동작을 검증한다.

모든 점검용 파일은 TemporaryDirectory 안에만 생성한다.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_assignment4 as assignment
from algorithms.ppo import PPO
from utils.logger import RunLogger


class AssignmentTests(unittest.TestCase):
    def test_sumo_connection_and_observation_without_model_training(self):
        from env import road_builder
        from env.sumo_env import SumoHighwayEnv
        from env.road_config import ROAD, EGO, TRAFFIC
        from env.mdp_config import SIMULATION, OBSERVATION, ACTION
        from train import REWARD
        with tempfile.TemporaryDirectory() as tmp:
            cfg = road_builder.build(ROAD, EGO, TRAFFIC, tmp)
            env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, traffic=TRAFFIC,
                                 mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                                 action=ACTION, reward=REWARD, gui=False)
            try:
                obs, _ = env.reset()
                self.assertEqual(obs.shape, (31,))
                self.assertTrue(np.isfinite(obs).all())
                obs, reward, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
            finally:
                env.close()

    def test_checkpoint_restores_policy_optimizer_steps_and_rng(self):
        config = {"model": "a", "hparams": {"lr": 1e-4}}
        agent = PPO(31, 2, hidden_sizes=(8, 8), device="cpu")
        agent.num_timesteps = 4096
        param = next(agent.policy.parameters())
        # 학습 없이 알려진 optimizer 상태를 직접 설정해 복원을 검증한다.
        agent.optimizer.state[param] = {
            "step": torch.tensor(3.), "exp_avg": torch.ones_like(param),
            "exp_avg_sq": torch.full_like(param, 2.),
        }
        saved_weights = {k: v.clone() for k, v in agent.policy.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            assignment.save_checkpoint(agent, run, config, {})
            expected_torch = torch.rand(4)
            expected_numpy = np.random.rand(4)
            restored = PPO(31, 2, hidden_sizes=(8, 8), device="cpu")
            assignment.restore_checkpoint(restored, run, config)
            self.assertEqual(restored.num_timesteps, 4096)
            for k, v in restored.policy.state_dict().items():
                self.assertTrue(torch.equal(v, saved_weights[k]))
            state = restored.optimizer.state[next(restored.policy.parameters())]
            self.assertEqual(state["step"].item(), 3)
            self.assertTrue(torch.equal(state["exp_avg"], torch.ones_like(param)))
            self.assertTrue(torch.equal(state["exp_avg_sq"], torch.full_like(param, 2)))
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            np.testing.assert_equal(np.random.rand(4), expected_numpy)
            with self.assertRaises(RuntimeError):
                assignment.restore_checkpoint(restored, run, {"model": "b"})

    def test_learn_callback_stops_at_boundary_and_preserves_update_count(self):
        # 실제 rollout 수집이나 optimizer 업데이트는 실행하지 않는다.
        agent = object.__new__(PPO)
        agent.n_steps = 2048
        agent.num_timesteps = 8 * 2048
        agent.gamma, agent.gae_lambda = .99, .95
        agent.buffer = SimpleNamespace(compute_gae=lambda *args: ([], []))
        agent.policy = SimpleNamespace(current_std=.5)
        agent.update = lambda *args: {}

        def collect(*args, **kwargs):
            agent.num_timesteps += 2048
            return np.zeros(31), 0., [1.]

        agent.collect_rollout = collect
        events = []
        logger = SimpleNamespace(
            log_train=lambda step, metrics, **kw: events.append(("train", metrics["update"])),
            log_eval=lambda *args: events.append(("eval", agent.num_timesteps)),
        )
        env = SimpleNamespace(reset=lambda: (np.zeros(31), {}))
        called = []

        def callback(current):
            events.append(("checkpoint", current.num_timesteps))
            called.append(current.num_timesteps)
            return len(called) < 2

        metrics = SimpleNamespace(as_dict=lambda: {}, summary=lambda: "mock")
        with patch("algorithms.ppo.evaluate_policy", return_value=metrics):
            agent.learn(env, 200000, logger=logger, eval_interval=5, on_update=callback)
        self.assertEqual(called, [9 * 2048, 10 * 2048])
        self.assertEqual([v for k, v in events if k == "train"], [9, 10])
        self.assertLess(events.index(("eval", 10 * 2048)),
                        events.index(("checkpoint", 10 * 2048)))

    def test_logger_resume_keeps_prior_rows_and_discards_uncommitted_rows(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(tmp)
            logger.log_train(2048, {"update": 1, "ep_rew_mean": 1.}, verbose=False)
            logger.log_train(4096, {"update": 2, "ep_rew_mean": 2.}, verbose=False)
            logger.log_episode(4096, 3., 10)
            logger.close()
            resumed = RunLogger(tmp, resume_step=2048)
            self.assertEqual(len(resumed.train_rows), 1)
            resumed.log_train(4096, {"update": 2, "ep_rew_mean": 4.}, verbose=False)
            resumed.close()
            events = EventAccumulator(str(Path(tmp) / "tb")).Reload()
            scalars = events.Scalars("train/ep_rew_mean")
            self.assertEqual([s.step for s in scalars], [2048, 4096])
            self.assertEqual([s.value for s in scalars], [1., 4.])
            self.assertEqual(events.Scalars("episode/return"), [])

    def test_assignment_new_models_resume_and_stop_without_training(self):
        from env import road_builder
        from env import sumo_env
        fake_env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(31,)),
            action_space=SimpleNamespace(shape=(2,)), close=lambda: None,
        )

        def no_training_learn(agent, env, total_timesteps, logger, on_update, **kwargs):
            # Synthetic counters only; no observations or gradient steps.
            while agent.num_timesteps < total_timesteps:
                agent.num_timesteps += agent.n_steps
                if on_update(agent) is False:
                    break

        def args(model, **overrides):
            return SimpleNamespace(model=model, new_run=False, timesteps=4096,
                                   session_updates=1, **overrides)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(assignment, "RESULTS", Path(tmp)), \
                patch.object(sumo_env, "SumoHighwayEnv", return_value=fake_env), \
                patch.object(road_builder, "build", return_value="mock.sumocfg"), \
                patch.object(PPO, "learn", no_training_learn), \
                contextlib.redirect_stdout(io.StringIO()):
            assignment.train(args("a"))
            run_a = assignment.latest_run("a")
            info = json.loads((run_a / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["status"], "paused")
            self.assertEqual(info["trained_steps"], 2048)
            original_start = info["started_at_kst"]
            assignment.train(args("b"))
            run_b = assignment.latest_run("b")
            self.assertNotEqual(run_a, run_b)
            config_a = json.loads((run_a / "experiment_config.json").read_text(encoding="utf-8"))
            config_b = json.loads((run_b / "experiment_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config_a["hparams"]["lr"], 1e-4)
            self.assertEqual(config_b["hparams"]["lr"], 3e-4)
            assignment.train(args("a"))
            self.assertEqual(assignment.latest_run("a"), run_a)
            info = json.loads((run_a / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["trained_steps"], 4096)
            self.assertEqual(info["status"], "complete")
            self.assertEqual(len(info["sessions"]), 2)
            self.assertEqual(info["started_at_kst"], original_start)
            assignment.train(args("a"))
            info = json.loads((run_a / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(len(info["sessions"]), 2)

    def test_interrupt_rolls_back_incomplete_update_and_purges_events(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        fake_env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(31,)),
            action_space=SimpleNamespace(shape=(2,)), close=lambda: None,
        )
        before = {}

        def interrupted_learn(agent, env, logger, on_update, **kwargs):
            before.update({k: v.clone() for k, v in agent.policy.state_dict().items()})
            agent.num_timesteps += agent.n_steps
            logger.log_train(agent.num_timesteps, {"update": 1}, verbose=False)
            on_update(agent)
            agent.num_timesteps += 123
            logger.log_episode(agent.num_timesteps, 2., 123)
            with torch.no_grad():
                next(agent.policy.parameters()).add_(9)
            raise KeyboardInterrupt

        args = SimpleNamespace(model="a", new_run=False, timesteps=4096, session_updates=10)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(assignment, "RESULTS", Path(tmp)), \
                patch("env.sumo_env.SumoHighwayEnv", return_value=fake_env), \
                patch("env.road_builder.build", return_value="mock.sumocfg"), \
                patch.object(PPO, "learn", interrupted_learn), \
                contextlib.redirect_stdout(io.StringIO()):
            assignment.train(args)
            run = assignment.latest_run("a")
            state = torch.load(run / "checkpoint.pt", weights_only=True)
            self.assertEqual(state["num_timesteps"], 2048)
            for k, value in state["policy"].items():
                self.assertTrue(torch.equal(value, before[k]))
            info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["status"], "paused")
            events = EventAccumulator(str(run / "tb")).Reload()
            self.assertEqual(events.Scalars("episode/return"), [])
            from test import load_agent
            restored = load_agent(str(run / "model.pt"), 31, 2, algorithm="ppo")
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], 1e-4)
            for k, value in restored.policy.state_dict().items():
                self.assertTrue(torch.equal(value, before[k]))


if __name__ == "__main__":
    unittest.main()

