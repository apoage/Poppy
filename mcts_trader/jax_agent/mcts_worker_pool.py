import multiprocessing as mp
import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Any
import numpy as np
import os
import sys

# Global variable to hold JIT-compiled search function in worker process
_WORKER_SEARCH_FN = None

def _init_worker(model_cls, model_kwargs: Dict[str, Any], search_kwargs: Dict[str, Any]):
    """
    Initializer for worker process.
    Sets up CPU-only environment, instantiates model, creates search policy, and JIT compiles it.
    """
    global _WORKER_SEARCH_FN
    
    # Hide GPU from workers
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Force JAX to use CPU backend (programmatic override)
    # This prevents workers from seeing GPU even if env var was set late
    jax.config.update("jax_platform_name", "cpu")
    
    # Pre-allocate some JAX memory? No, let JAX manage on CPU.
    # Improve CPU parallelism for JAX?
    # os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' 
    # (Maybe, to avoid oversubscription if we run many workers)
    
    try:
        from mcts_trader.jax_agent.mctx_search import create_mctx_search
        
        # Instantiate Model
        # We assume model_cls is passed or importable. Passing class object works if picklable.
        model = model_cls(**model_kwargs)
        
        # Create Search Policy
        # search_kwargs should match arguments of create_mctx_search
        # (model, num_simulations, use_gumbel, etc.)
        # Note: create_mctx_search expects 'model' as first arg
        search_fn = create_mctx_search(model, **search_kwargs)
        
        # JIT Compile
        _WORKER_SEARCH_FN = jax.jit(search_fn)
        
        # Warmup? Optional.
        
    except Exception as e:
        print(f"Error initializing worker: {e}", file=sys.stderr)
        raise e

def _worker_task(params, rng, obs, masks):
    """
    Task function executed by worker.
    """
    global _WORKER_SEARCH_FN
    if _WORKER_SEARCH_FN is None:
        raise RuntimeError("Worker not initialized!")
        
    # params comes in as numpy arrays (pickled), JAX automatically handles them?
    # Yes, JAX can ingest numpy arrays.
    
    # JAX operations
    return _WORKER_SEARCH_FN(params, rng, obs, masks)


class MCTSWorkerPool:
    """Pool of MCTS workers for parallel search"""

    def __init__(self, model_cls, model_kwargs: Dict, search_kwargs: Dict, n_workers: int = 4):
        """
        Args:
            model_cls: MuZeroNetworks class (picklable)
            model_kwargs: Dict of args for model init
            search_kwargs: Dict of args for create_mctx_search
            n_workers: Number of workers
        """
        self.n_workers = n_workers
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.search_kwargs = search_kwargs

        # Create worker pool with initializer
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass 
            
        self.pool = mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(model_cls, model_kwargs, search_kwargs)
        )
        
        # Create local search function for small batches (fallback)
        # This avoids IPC overhead for sequential self-play (batch=1)
        try:
            from mcts_trader.jax_agent.mctx_search import create_mctx_search
            local_model = model_cls(**model_kwargs)
            local_search = create_mctx_search(local_model, **search_kwargs)
            self.local_search_fn = jax.jit(local_search)
            print("✅ Local JIT fallback created for small batches")
        except Exception as e:
            print(f"⚠️ Failed to create local fallback: {e}")
            self.local_search_fn = None

        print(f"✅ MCTS Worker Pool initialized with {n_workers} workers")

    def __call__(
        self,
        params,
        rng,
        observations,
        action_masks
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """
        Run MCTS in parallel across workers.
        """
        batch_size = observations.shape[0]
        
        # Handle small batches: Run locally if possible, or on single worker
        if batch_size < self.n_workers:
            if self.local_search_fn is not None:
                return self.local_search_fn(params, rng, observations, action_masks)
            
            # Fallback to single worker if local failed (IPC overhead!)
            obs_np = np.array(observations)
            mask_np = np.array(action_masks)
            tasks = [(params, rng, obs_np, mask_np)]
            results = self.pool.starmap(_worker_task, tasks)
            return results[0][0], results[0][1], {}

        chunk_size = batch_size // self.n_workers
        
        # Prepare inputs (convert to numpy to ensure picklability and avoid GPU-pointer issues)
        obs_np = np.array(observations)
        mask_np = np.array(action_masks)
        
        rng_keys = jax.random.split(rng, self.n_workers)

        tasks = []
        for i in range(self.n_workers):
            start = i * chunk_size
            # Ensure the last worker gets the remainder
            end = start + chunk_size if i < self.n_workers - 1 else batch_size
            
            obs_chunk = obs_np[start:end]
            mask_chunk = mask_np[start:end]
            
            tasks.append((params, rng_keys[i], obs_chunk, mask_chunk))

        # Run
        results = self.pool.starmap(_worker_task, tasks)

        # Gather
        all_actions = []
        all_policies = []

        for actions, policies, _ in results:
            all_actions.append(actions)
            all_policies.append(policies)

        # Concatenate
        actions = jnp.concatenate(all_actions, axis=0)
        policies = jnp.concatenate(all_policies, axis=0)

        # Stats? Merging stats is tricky (e.g. mean value). Ignore for now.
        return actions, policies, {}

    def close(self):
        self.pool.close()
        self.pool.join()