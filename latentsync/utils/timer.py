import time
from functools import wraps
from typing import Dict, Optional

class Timer:
    """用于统计函数运行时间的装饰器类"""
    
    _stats: Dict[str, Dict[str, float]] = {}
    
    def __init__(self, name: Optional[str] = None, print_args: bool = False):
        """
        Args:
            name: 计时器名称，如果不提供则使用被装饰函数的名称
            print_args: 是否打印函数参数
        """
        self.name = name
        self.print_args = print_args
    
    def __call__(self, func):
        if self.name is None:
            self.name = func.__name__
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 开始计时
            start_time = time.time()
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            
            # 更新统计信息
            if self.name not in Timer._stats:
                Timer._stats[self.name] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': float('-inf')
                }
            
            stats = Timer._stats[self.name]
            stats['count'] += 1
            stats['total_time'] += elapsed_time
            stats['min_time'] = min(stats['min_time'], elapsed_time)
            stats['max_time'] = max(stats['max_time'], elapsed_time)
            
            # 打印信息
            args_str = f"args: {args}, kwargs: {kwargs}" if self.print_args else ""
            print(f"[Timer] {self.name} {args_str} - 耗时: {elapsed_time*1000:.2f}ms")
            
            return result
            
        return wrapper
    
    @staticmethod
    def print_stats():
        """打印所有计时器的统计信息"""
        print("\n=== Timer Statistics ===")
        for name, stats in Timer._stats.items():
            avg_time = stats['total_time'] / stats['count']
            print(f"\n{name}:")
            print(f"  调用次数: {stats['count']}")
            print(f"  平均耗时: {avg_time*1000:.2f}ms")
            print(f"  最小耗时: {stats['min_time']*1000:.2f}ms")
            print(f"  最大耗时: {stats['max_time']*1000:.2f}ms")
            print(f"  总耗时: {stats['total_time']*1000:.2f}ms")
    
    @staticmethod
    def reset_stats():
        """重置所有计时器的统计信息"""
        Timer._stats.clear() 