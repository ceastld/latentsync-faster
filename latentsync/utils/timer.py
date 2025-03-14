import time
from functools import wraps
from typing import Dict, Optional, List

class Timer:
    """A decorator class for measuring function execution time"""
    
    _stats: Dict[str, Dict[str, float | List[float]]] = {}
    _enabled: bool = False
    
    def __init__(self, name: Optional[str] = None, print_args: bool = False):
        """
        Args:
            name: Timer name, uses decorated function name if not provided
            print_args: Whether to print function arguments
        """
        self.name = name
        self.print_args = print_args
    
    def __call__(self, func):
        if self.name is None:
            self.name = func.__name__
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not Timer._enabled:
                return func(*args, **kwargs)
                
            start_time = time.time()
            result = func(*args, **kwargs)
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
    
    @staticmethod
    def print_stats():
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
            print(f"  Avg time: {(stats['total_time']/stats['count'])*1000:.2f}ms")
            print(f"  Min time: {stats['min_time']*1000:.2f}ms")
            print(f"  Max time: {stats['max_time']*1000:.2f}ms")
            print(f"  Total time: {stats['total_time']*1000:.2f}ms")
    
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