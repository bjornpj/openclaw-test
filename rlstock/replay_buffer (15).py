#code from openai
#https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import random
import torch as T
import operator
import itertools

from math import sqrt
from collections import deque
from typing import Deque, Dict, List, Tuple, Callable

# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

##############################################################
import numpy as np
import torch as T
from collections import deque
from math import sqrt
from typing import Tuple, Dict
from numba import njit

@njit
def find_prefix_sum_idx_numba(prefix_sums, priority_sum, capacity):
    idxs = np.empty(len(prefix_sums), dtype=np.int32)
    for i in range(len(prefix_sums)):
        prefix_sum = prefix_sums[i]
        idx = 1
        while idx < capacity:
            left = idx << 1
            if priority_sum[left] > prefix_sum:
                idx = left
            else:
                prefix_sum -= priority_sum[left]
                idx = left + 1
        idxs[i] = idx - capacity
    return idxs

@njit
def compute_discounted_reward(rews, gamma):
    discounts = np.power(np.float32(gamma), np.arange(len(rews), dtype=np.float32))
    return np.dot(rews, discounts)

@njit
def batch_set_priorities(priority_sum, priority_min, indices, priorities, capacity):
    for i in range(len(indices)):
        idx = indices[i] + capacity
        priority_sum[idx] = priorities[i]
        priority_min[idx] = priorities[i]

    for i in range(len(indices)):
        idx = indices[i] + capacity
        while idx > 1:
            idx //= 2
            left = idx * 2
            right = left + 1
            priority_sum[idx] = priority_sum[left] + priority_sum[right]
            priority_min[idx] = min(priority_min[left], priority_min[right])

@njit
def build_segment_tree(priorities, priority_sum, priority_min, capacity):
    for i in range(len(priorities)):
        idx = i + capacity
        priority_sum[idx] = priorities[i]
        priority_min[idx] = priorities[i]
    for i in range(capacity - 1, 0, -1):
        left = 2 * i
        right = left + 1
        priority_sum[i] = priority_sum[left] + priority_sum[right]
        priority_min[i] = min(priority_min[left], priority_min[right])

@njit
def reset_segment_tree(priority_sum, priority_min, capacity):
    for i in range(len(priority_sum)):
        priority_sum[i] = 0.0
        priority_min[i] = np.inf
    priority_sum[0] = 0.0
    priority_min[0] = np.inf


class PrioritizedReplayBufferNew:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 64, alpha: float = 0.6,
                 b_step: int = 0, n_step: int = 1, gamma: float = 0.99,
                 nenvs=64, ndays=16, dev=None):
        self.capacity = size
        self.gamma = gamma
        self.n_step = n_step
        self.n_envs = nenvs
        self.n_step_buffers = [deque(maxlen=n_step) for _ in range(nenvs)]

        self.priority_sum = np.zeros(2 * size)
        self.priority_min = np.full(2 * size, np.inf)
        self.max_priority = 1.0

        self.next_idx = 0
        self.size = 0
        self.device = dev

#        self.obs_buf = T.zeros([size, 20, ndays + 1, 1], dtype=T.float32, device=self.device)
#        self.next_obs_buf = T.zeros([size, 20, ndays + 1, 1], dtype=T.float32, device=self.device)
        self.obs_buf = T.zeros([size, 19, ndays + 1, 1], dtype=T.float32, device=self.device)
        self.next_obs_buf = T.zeros([size, 19, ndays + 1, 1], dtype=T.float32, device=self.device)
#1000        self.obs_buf = T.zeros([size, 16, ndays + 1, 1], dtype=T.float32, device=self.device)
#1000        self.next_obs_buf = T.zeros([size, 16, ndays + 1, 1], dtype=T.float32, device=self.device)
        
        # one-time memory layout (fast for Conv2d)
#        self.obs_buf = self.obs_buf.to(memory_format=T.channels_last)
#        self.next_obs_buf = self.next_obs_buf.to(memory_format=T.channels_last)
        
        self.acts_buf = T.zeros([size, 1], dtype=T.int64, device=self.device)
        self.rews_buf = T.zeros([size, 1], dtype=T.float32, device=self.device)
        self.done_buf = T.zeros([size, 1], dtype=T.bool, device=self.device)

        self.max_size = size
        self.batch_size = batch_size

        self.alpha = float(alpha)   # PER exponent (0.5 -> sqrt, 0.6 default, etc.)
        self.eps_pri = 1e-6         # small floor to avoid zero priority
#805        self.alpha = 0.01

    def _to_alpha(self, p):
        """Raise raw TD-error priorities to alpha with a small floor."""
        p = np.asarray(p, dtype=np.float32)
        p = np.maximum(p, self.eps_pri)
        if self.alpha == 0.5:
            return np.sqrt(p)           # fast path for alpha=0.5
        return np.power(p, self.alpha)  # general case

    def store_new(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray,
              done: bool, info, j: int) -> Tuple:
    
        done_union = bool(done)
        term_flag  = bool(info.get("terminated", done_union)) if isinstance(info, dict) else done_union
    
        transition = (obs, act, rew, next_obs, done_union, term_flag)
        self.n_step_buffers[j].append(transition)
    
        if len(self.n_step_buffers[j]) < self.n_step or self.n_step_buffers[j][0][4]:
            return ()
    
        obs0, act0 = self.n_step_buffers[j][0][:2]
        rew_n, next_obs_n, done_term_only = self._get_n_step_info(self.n_step_buffers[j], self.gamma)
    
        idx = self.next_idx
    
        # ---- fast copies (handles np.ndarray or torch.Tensor) ----
        if isinstance(obs0, T.Tensor):
            # If already a tensor, move with non_blocking if on CPU
            if obs0.device.type == "cpu":
                self.obs_buf[idx].copy_(obs0.pin_memory().to(self.device, non_blocking=True))
            else:
                self.obs_buf[idx].copy_(obs0)
        else:
            # NumPy → pinned CPU tensor → async H2D
            cpu_obs = T.from_numpy(obs0).to(dtype=T.float32).pin_memory()
            self.obs_buf[idx].copy_(cpu_obs.to(self.device, non_blocking=True))
    
        if isinstance(next_obs_n, T.Tensor):
            if next_obs_n.device.type == "cpu":
                self.next_obs_buf[idx].copy_(next_obs_n.pin_memory().to(self.device, non_blocking=True))
            else:
                self.next_obs_buf[idx].copy_(next_obs_n)
        else:
            cpu_next = T.from_numpy(next_obs_n).to(dtype=T.float32).pin_memory()
            self.next_obs_buf[idx].copy_(cpu_next.to(self.device, non_blocking=True))
    
        # scalars / small tensors
        self.acts_buf[idx, 0] = act0
        self.rews_buf[idx, 0] = rew_n
        self.done_buf[idx, 0] = bool(done_term_only)
    
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
    
        priority = self._to_alpha(self.max_priority)
        batch_set_priorities(self.priority_sum, self.priority_min,
                             np.array([idx]), np.array([priority]), self.capacity)
    
        return self.n_step_buffers[j][0]

    
    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray,
              done: bool, info, j: int) -> Tuple:        
        # Extract flags
        done_union = bool(done)
        term_flag = bool(info.get("terminated", done_union)) if isinstance(info, dict) else bool(done_union)
#        term_flag  = bool(getattr(info, "get", lambda *_: False)("terminated", done_union)) 
        
        # Pack both flags in the transition:
        #   x[4] = done_union  (used to respect boundaries in your existing logic)
        #   x[5] = term_flag   (we will use this to make the loss mask termination-only)
        transition = (obs, act, rew, next_obs, done_union, term_flag)
        self.n_step_buffers[j].append(transition)

        if len(self.n_step_buffers[j]) < self.n_step or self.n_step_buffers[j][0][4]:
            return ()

        obs, act = self.n_step_buffers[j][0][:2]
        rew, next_obs, done_term_only = self._get_n_step_info(self.n_step_buffers[j], self.gamma)

        idx = self.next_idx
        self.obs_buf[idx].copy_(T.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device))
        self.next_obs_buf[idx].copy_(T.from_numpy(np.asarray(next_obs, dtype=np.float32)).to(self.device))
        self.acts_buf[idx, 0] = act
        self.rews_buf[idx, 0] = rew
        # ↓↓↓ store **termination-only** into done_buf
        self.done_buf[idx, 0] = bool(done_term_only)

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        priority = self._to_alpha(self.max_priority)
#z804        priority = sqrt(self.max_priority)
        batch_set_priorities(self.priority_sum, self.priority_min, 
                             np.array([idx]), np.array([priority]), 
                             self.capacity)

        return self.n_step_buffers[j][0]

    def _get_n_step_info(self, buf, gamma):
        """
        buf[i] layout:
          0: obs, 1: act, 2: rew, 3: next_obs, 4: done_union, 5: term_flag

        Logic:
          - accumulate discounted rewards up to n steps
          - stop at the *first* boundary (terminated OR truncated)
          - return:
              R_n       = discounted sum up to that boundary (or n steps)
              next_obs  = obs after the last reward we used
              done_term = True IFF a *true* terminal was encountered
        """
        R = 0.0
        discount = 1.0
        next_obs = buf[-1][3]   # default: last transition's next_obs
        done_term_only = False

        for i in range(len(buf)):
            r = float(buf[i][2])
            R += discount * r
            discount *= gamma

            # boundary: either terminated or truncated
            if buf[i][4]:
                next_obs = buf[i][3]
                done_term_only = bool(buf[i][5])  # True only for real terminal
                break

        return R, next_obs, done_term_only

    
    def _get_n_step_info_old(self, buf, gamma):
        """
        buf[i] layout (after the minimal patch):
          0: obs, 1: act, 2: rew, 3: next_obs, 4: done_union, 5: term_flag
        We:
          - discount-sum rewards over the current buffer
          - stop on first boundary (union) to avoid crossing episodes
          - return TERMINATION-ONLY as 'done' (term_flag), so loss bootstraps on truncations
        """
        rews = np.array([x[2] for x in buf], dtype=np.float32)
        # your helper; alternatively inline dot(rews, gamma**arange)
        rew = compute_discounted_reward(rews, gamma)
    
        for x in buf:
            if x[4]:  # boundary: either terminated OR truncated
                return rew, x[3], bool(x[5])  # done = term_flag only
    
        # no boundary inside the window
        return rew, buf[-1][3], False
    

    def _set_priority(self, idx, priority):
        priority_idx = idx + self.capacity
        self.priority_min[priority_idx] = priority
        self.priority_sum[priority_idx] = priority

        while priority_idx > 1:
            priority_idx //= 2
            left = priority_idx * 2
            right = left + 1
            self.priority_min[priority_idx] = min(self.priority_min[left], self.priority_min[right])
            self.priority_sum[priority_idx] = self.priority_sum[left] + self.priority_sum[right]

    @T.no_grad()
    def sample_batch(self, beta: float = 0.4, batch_size: int = 64) -> Dict[str, T.Tensor]:
        """
        Sample a PER batch without pin_memory / async transfer overhead.
        All replay storage (obs_buf, etc.) is already on self.device.
        """
        # --- draw prefix-sum targets in [0, total_priority] ---
        total_priority = self.priority_sum[1]
        p = np.random.uniform(0.0, total_priority, size=batch_size)

        # numba-accelerated search over segment tree
        indices = find_prefix_sum_idx_numba(p, self.priority_sum, self.capacity)

        # --- importance-sampling weights ---
        leaf_priorities = self.priority_sum[indices + self.capacity]  # [batch]
        prob = leaf_priorities / total_priority                       # P(i)

        prob_min = self.priority_min[1] / total_priority
        max_weight = (prob_min * self.size) ** (-beta)
        weights_np = (prob * self.size) ** (-beta) / max_weight       # [batch]

        # move small arrays directly to device (no pinning / async)
        idx_t = T.as_tensor(indices, dtype=T.long, device=self.device)
        weights = T.as_tensor(weights_np, dtype=T.float32, device=self.device)

        # --- gather batch from pre-allocated device buffers ---
        obs      = self.obs_buf.index_select(0, idx_t)
        next_obs = self.next_obs_buf.index_select(0, idx_t)
        acts     = self.acts_buf.index_select(0, idx_t)
        rews     = self.rews_buf.index_select(0, idx_t)
        done     = self.done_buf.index_select(0, idx_t)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "acts": acts,
            "rews": rews,
            "done": done,
            "weights": weights,
            "indices": indices,   # keep as np.ndarray for PER updates
        }

    
    @T.no_grad()
    def sample_batch_120625(self, beta: float = 0.4, batch_size: int = 64) -> Dict[str, T.Tensor]:
        # Pre-allocate memory for numpy arrays
        p = np.random.uniform(0, self.priority_sum[1], size=batch_size)
        
        # Use numba function for better performance
        indices = find_prefix_sum_idx_numba(p, self.priority_sum, self.capacity)
        
        # Vectorized operations instead of loops
        prob = self.priority_sum[indices + self.capacity] / self.priority_sum[1]
        
        # Fuse computations when possible
        prob_min = self.priority_min[1] / self.priority_sum[1]
        max_weight = (prob_min * self.size) ** (-beta)
        weights_np = (prob * self.size) ** (-beta) / max_weight
        
        # Move data to GPU in parallel with computation
        # Use CUDA streams for asynchronous transfer when available
        weights = T.from_numpy(weights_np.astype(np.float32)).pin_memory().to(self.device, non_blocking=True)
        idx_t = T.from_numpy(indices).pin_memory().to(self.device, non_blocking=True)
        
        # Prefetch tensors in parallel for better memory access patterns
        obs = self.obs_buf.index_select(0, idx_t)
        next_obs = self.next_obs_buf.index_select(0, idx_t)
        acts = self.acts_buf.index_select(0, idx_t)
        rews = self.rews_buf.index_select(0, idx_t)
        done = self.done_buf.index_select(0, idx_t)
        
        return {
            "obs": obs,
            "next_obs": next_obs,
            "acts": acts,
            "rews": rews,
            "done": done,
            "weights": weights,
            "indices": indices
        }

    def update_priorities(self, indexes, priorities):
        priorities = np.asarray(priorities, dtype=np.float32).flatten()
        self.max_priority = max(self.max_priority, float(priorities.max()))
        priorities_alpha = self._to_alpha(priorities)
#z804        priorities = np.asarray(priorities).flatten()
#z804        priorities_alpha = np.sqrt(priorities)
#z804        self.max_priority = max(self.max_priority, priorities.max())

        batch_set_priorities(self.priority_sum, self.priority_min,
                             np.asarray(indexes), priorities_alpha, self.capacity)

    @property
    def is_full(self):
        return self.capacity == self.size

    def __len__(self):
        return self.size