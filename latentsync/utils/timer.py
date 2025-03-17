import time
from functools import wraps
from typing import Dict, Optional, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class Timer:
    """
    A decorator class for measuring function execution time.
    
    This class provides accurate timing for both CPU and GPU operations.
    When timing PyTorch GPU operations, it can synchronize CUDA to ensure
    accurate measurements of asynchronous GPU operations.
    """
    
    _stats: Dict[str, Dict[str, float | List[float]]] = {}
    _enabled: bool = False
    
    def __init__(self, name: Optional[str] = None, print_args: bool = False, use_cuda: bool = True):
        """
        Args:
            name: Timer name, uses decorated function name if not provided
            print_args: Whether to print function arguments
            use_cuda: Whether to synchronize CUDA operations before timing.
                      When True and PyTorch is available, calls torch.cuda.synchronize()
                      before and after the timed operation to ensure accurate
                      measurement of asynchronous GPU operations.
        """
        self.name = name
        self.print_args = print_args
        self.use_cuda = use_cuda and TORCH_AVAILABLE
        self.start_time = None
    
    def __call__(self, func):
        if self.name is None:
            self.name = func.__name__
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not Timer._enabled:
                return func(*args, **kwargs)
            
            if self.use_cuda:
                torch.cuda.synchronize()
                
            start_time = time.time()
            result = func(*args, **kwargs)
            
            if self.use_cuda:
                torch.cuda.synchronize()
                
            elapsed_time = time.time() - start_time
            
            if self.name not in Timer._stats:
                Timer._stats[self.name] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': float('-inf'),
                    'time_records': []
                }
            
            stats = Timer._stats[self.name]
            stats['count'] += 1
            stats['total_time'] += elapsed_time
            stats['min_time'] = min(stats['min_time'], elapsed_time)
            stats['max_time'] = max(stats['max_time'], elapsed_time)
            stats['time_records'].append(elapsed_time)
            
            args_str = f"args: {args}, kwargs: {kwargs}" if self.print_args else ""
            print(f"[Timer] {self.name} {args_str} - Time: {elapsed_time*1000:.2f}ms")
            
            return result
            
        return wrapper
    
    def __enter__(self):
        """支持上下文管理器协议的入口方法"""
        if not Timer._enabled:
            return self
            
        if self.use_cuda:
            torch.cuda.synchronize()
            
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器协议的退出方法"""
        if not Timer._enabled or self.start_time is None:
            return
        
        if self.use_cuda:
            torch.cuda.synchronize()
            
        elapsed_time = time.time() - self.start_time
        
        if self.name not in Timer._stats:
            Timer._stats[self.name] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': float('-inf'),
                'time_records': []
            }
        
        stats = Timer._stats[self.name]
        stats['count'] += 1
        stats['total_time'] += elapsed_time
        stats['min_time'] = min(stats['min_time'], elapsed_time)
        stats['max_time'] = max(stats['max_time'], elapsed_time)
        stats['time_records'].append(elapsed_time)
        
        print(f"[Timer] {self.name} - Time: {elapsed_time*1000:.2f}ms")
    
    @staticmethod
    def summary():
        """Print statistics for all timers"""
        if not Timer._enabled:
            return
            
        print("\n=== Timer Statistics ===")
        for name, stats in Timer._stats.items():
            time_records = stats['time_records']
            
            trimmed_mean = None
            if len(time_records) > 2:
                sorted_times = sorted(time_records)
                trimmed_times = sorted_times[1:-1]
                trimmed_mean = sum(trimmed_times) / len(trimmed_times)
            
            print(f"\n{name}:")
            print(f"  Calls: {stats['count']}")
            if trimmed_mean is not None:
                print(f"  Avg time (trimmed): {trimmed_mean*1000:.2f}ms")
            else:
                print(f"  Avg time: {(stats['total_time']/stats['count'])*1000:.2f}ms")
            # print(f"  Min time: {stats['min_time']*1000:.2f}ms")
            # print(f"  Max time: {stats['max_time']*1000:.2f}ms")
            # print(f"  Total time: {stats['total_time']*1000:.2f}ms")
    
    @staticmethod
    def reset_stats():
        """Reset all timer statistics"""
        Timer._stats.clear()
        
    @staticmethod
    def enable():
        """Enable timer"""
        Timer._enabled = True
        
    @staticmethod
    def disable():
        """Disable timer"""
        Timer._enabled = False
        
    @staticmethod
    def is_enabled() -> bool:
        """Return whether timer is enabled"""
        return Timer._enabled 