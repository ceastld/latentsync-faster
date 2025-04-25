from typing import AsyncGenerator, List, TypeVar, Generic, Optional
from latentsync.inference.multi_infer import MultiThreadInference

T = TypeVar('T')  # Type for buffer items
R = TypeVar('R')  # Type for inference results

class BufferInference(MultiThreadInference, Generic[T, R]):
    """Base class for buffer-based inference.
    
    This class provides common functionality for inference tasks that need to buffer
    data before processing in batches.
    """
    
    def __init__(self, batch_size: int, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)
        self.batch_size = batch_size
        self.data_buffer: List[T] = []
        
    def push_data(self, data: T):
        """Push single data item to buffer."""
        self.data_buffer.append(data)
        if len(self.data_buffer) >= self.batch_size:
            self.add_one_task(self.data_buffer)
            self.data_buffer = []
            
    def push_data_batch(self, data: List[T]):
        """Push multiple data items to buffer."""
        self.data_buffer.extend(data)
        if len(self.data_buffer) >= self.batch_size:
            self.add_one_task(self.data_buffer)
            self.data_buffer = []
            
    def add_end_task(self):
        """Process remaining data in buffer before ending."""
        if len(self.data_buffer) > 0:
            self.add_one_task(self.data_buffer)
            self.data_buffer = []
        super().add_end_task()
        
    def infer_task(self, model, data: List[T]) -> List[R]:
        """Process a batch of data.
        
        This method should be implemented by subclasses to define how to process
        the batched data using the provided model.
        """
        raise NotImplementedError("Subclasses must implement infer_task") 
    
    async def result_stream(self) -> AsyncGenerator[R, None]:
        """Stream results as they are available."""
        results: List[R] 
        async for results in super().result_stream():
            for result in results:
                yield result
