
import torch
import numpy as np
import random
from src.rl.policy_meta import MetaPolicy
from src.rl.buffer_adapt import AdaptationBuffer
from src.sim.event_bus import sim_bus
from src.sim.exchange import exchange_sim
from src.sim.scenario_presets import SCENARIOS

# v42 Meta-RL: MAML Trainer
# Orchestrates Inner Loop (Adaptation) and Outer Loop (Meta-Update) using v41 Simulator.

class MetaTrainer:
    def __init__(self, obs_dim, act_dim, tasks=None, inner_lr=0.01, outer_lr=3e-4):
        self.policy = MetaPolicy(obs_dim, act_dim, lr=outer_lr, inner_lr=inner_lr)
        self.buffer = AdaptationBuffer()
        self.tasks = tasks or list(SCENARIOS.keys())
        self.inner_steps = 1 # Number of adaptation gradient steps

    def sample_task(self):
        """Samples a random task (scenario) and returns its factory."""
        task_name = random.choice(self.tasks)
        return task_name, SCENARIOS[task_name]

    def collect_trajectory(self, policy, task_factory, duration=10.0):
        """Run one episode in the simulator with the given policy."""
        # Reset Sim
        sim_bus.reset()
        exchange_sim.book = exchange_sim.book.__class__()
        
        # Setup Agents from Task
        agents = task_factory()
        for agent in agents:
            agent.start()

        # Run Sim
        steps = 0
        rewards = []
        obs = np.random.randn(8) # Placeholder Obs (Sim doesn't emit full RL obs yet - Stub)
        
        # Stub Loop: In real RL integration, we'd hook into Sim step-by-step
        # Here we simulate an episode loop
        while steps < duration:
            action, logp = policy.get_action(obs)
            
            # Executing Action in Sim (Stub - assume Action is Price Offset)
            # In full impl, this would place an order
            
            # Next Step
            next_obs = np.random.randn(8)
            reward = random.uniform(-1, 1) # Stub Reward
            value = 0.0 # Stub Value
            
            self.buffer.store(obs, action, reward, value, logp)
            
            obs = next_obs
            steps += 1
            rewards.append(reward)
            
        return sum(rewards)

    def meta_update(self, meta_batch_size=4):
        """Performs one step of MAML meta-optimization."""
        meta_loss = 0.0
        
        for _ in range(meta_batch_size):
            task_name, task_factory = self.sample_task()
            
            # 1. Inner Loop (Support Set)
            train_policy = self.policy.clone()
            
            # Collect Support Data
            self.buffer.reset()
            self.buffer.set_mode("support")
            self.collect_trajectory(train_policy, task_factory, duration=10)
            
            # Adapt (Inner Gradient Step)
            batch = self.buffer.get_batch("support")
            
            # Adapt (Inner Gradient Step)
            batch = self.buffer.get_batch("support")
            
            # Compute PPO Loss (Simplified)
            # Re-evaluate to get gradients
            _, logp_new, _ = train_policy.evaluate(batch['obs'], batch['act'])
            
            # Simple PG Loss: - log_prob * advantage
            loss_inner = -(logp_new.squeeze() * batch['adv']).mean()
            
            train_policy.adapt(loss_inner)
            
            # 2. Outer Loop (Query Set)
            self.buffer.set_mode("query")
            # Run Adapted Policy on SAME task
            self.collect_trajectory(train_policy, task_factory, duration=10)
            
            batch_query = self.buffer.get_batch("query")
            
            # Calculate Meta-Loss on Main Policy Parameters (using adapted trajectory)
            _, logp_query, _ = train_policy.evaluate(batch_query['obs'], batch_query['act'])
            
            # Simple PG Loss
            loss_outer = -(logp_query.squeeze() * batch_query['adv']).mean()
            meta_loss += loss_outer
            
        # Meta Update
        meta_loss /= meta_batch_size
        self.policy.optimizer.zero_grad()
        meta_loss.backward()
        self.policy.optimizer.step()
        
        return meta_loss.item()
