import traceback
from typing import List
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Value, Lock, Manager
from tqdm import tqdm
import time
import torch
import asyncio
import asyncio
import warnings
import logging

class InferenceWorker:
    def __init__(self, num_workers=1, worker_timeout=60):
        self.worker_timeout = worker_timeout
        self.task_queue = mp.Queue()  # Queue to hold tasks dynamically
        manager = Manager()
        # self.results = manager.list()
        self.results = manager.dict()
        self.task_complete_counter = Value("i", 0)  # Shared variable to track progress.
        self.lock = Lock()
        self.num_workers = num_workers
        self.task_start_counter = Value("i", 0)
        self.task_wait_counter = Value("i", 0)
        pass

    def infer_task(self, model, input_data):
        """
        Inference task to be implemented by subclasses.
        :param model: The model to use for inference.
        :param input_data: The input data for inference.
        :return: The inference result, should be a dictionary.
        """
        model, input_data
        raise NotImplementedError("Subclasses must implement this method.")

    def _set_result(self, idx, result: np.ndarray):
        """
        Set the result of an inference task.
        :param idx: The index of the task.
        :param result: The result of the inference task, all value should be numpy.ndarray.
        """
        self.results[idx] = result
        self.task_complete_counter.value += 1

    def is_alive(self):
        raise NotImplementedError("Subclasses must implement this method.")

    async def _wait_for_id(self, id, remove=True):
        while self.is_alive():
            if id in self.results:
                if remove:
                    return self.results.pop(id)
                else:
                    return self.results[id]
            await asyncio.sleep(0.04)
        return None

    async def wait_one_result(self, remove=True):
        """
        Wait for a single result to be available.
        :param remove: Whether to remove the result after getting it.
        return: The result of the inference task, all value should be numpy.ndarray.
        """
        with self.lock:
            wait_idx = self.task_wait_counter.value
            self.task_wait_counter.value += 1
        return await self._wait_for_id(wait_idx, remove)
    
    async def result_stream(self):
        while self.is_alive() or len(self.results) > 0:
            result = await self.wait_one_result()
            yield result
        # for k in sorted(self.results.keys()):
        #     yield self.results[k]

    async def wait_for_results(self, count: int, pbar: tqdm = None, remove=True):
        """
        Wait for a specific number of results to be available.
        :param count: The number of results to wait for.
        :param pbar: Optional progress bar to update.
        :param remove: Whether to remove the result after getting it.
        """
        results = []
        with self.lock:
            wait_idx = self.task_wait_counter.value
            self.task_wait_counter.value += count
        ids = list(range(wait_idx, wait_idx + count))
        for id in ids:
            result = await self._wait_for_id(id, remove)
            results.append(result)
            if pbar:
                pbar.update(1)
        return results

    def add_one_task(self, task):
        with self.lock:
            start_idx = self.task_start_counter.value
            self.task_start_counter.value += 1
        self.task_queue.put((start_idx, task))

    def add_tasks(self, tasks):
        """
        Add tasks to the queue.
        :param tasks: A list of tasks to process. Each task can be a tuple or dictionary.
        :return: A range of task IDs that were added.
        """
        with self.lock:
            start_idx = self.task_start_counter.value
            self.task_start_counter.value += len(tasks)

        for idx, task in enumerate(tasks):
            self.task_queue.put((start_idx + idx, task))
        return range(start_idx, start_idx + len(tasks))

    def add_end_task(self):
        for _ in range(self.num_workers):
            self.task_queue.put((None, None))

    async def run(self, tasks):
        """
        Run inference tasks using multiple processes.
        :param tasks: A list of tasks to process. Each task can be a tuple or dictionary.
        :return: A list of inference results.
        """
        ids = self.add_tasks(tasks)
        return await self.wait_for_results(len(ids))

    def worker(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def start_workers(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def stop_workers(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def dispose(self):
        self.stop_workers()


class MultiThreadInference(InferenceWorker):
    def __init__(self, num_workers=1, worker_timeout=60):
        super().__init__(worker_timeout)
        self.num_workers = num_workers


class MultiProcessInference(InferenceWorker):
    mp.set_start_method("spawn", force=True)  # Use 'spawn' start method for CUDA.
    warnings.filterwarnings("ignore", category=FutureWarning)
    logger = logging.getLogger("MultiProcessInference")
    def __init__(self, num_workers=1, worker_timeout=60):
        """
        Initialize the multi-process inference class.
        should use `mp.set_start_method("spawn", force=True)` before using this class.
        :param num_processes: Number of processes to spawn.
        :param worker_timeout: Timeout for worker processes, in seconds.
        """
        super().__init__(num_workers=num_workers,worker_timeout=worker_timeout)
        self.worker_list: List[mp.Process] = []
        self.worker_loaded_count = Value("i", 0)
        self.worker_loaded_event = mp.Event()

    def get_model(self):
        """
        Load the model for inference. This is a placeholder method.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def wait_worker_loaded(self, timeout=None):
        if self.worker_loaded_count.value > 0:
            return True
        loaded = self.worker_loaded_event.wait(timeout=timeout)
        print("Worker Loaded")
        return loaded
    
    def process_task(self, model, idx, input_data):
        result = None
        if input_data is not None:
            result = self.infer_task(model, input_data)
        self._set_result(idx, result)
    
    def preprocess(self):
        """
        Preprocess function to be run before processing tasks
        """
        pass
    
    def worker(self):
        """
        Worker process function to process tasks from the queue.
        """
        model = self.get_model()
        self.worker_loaded_count.value += 1
        self.worker_loaded_event.set()
        while True:
            self.preprocess()
            idx, input_data = self.task_queue.get(timeout=self.worker_timeout)  # Timeout to prevent hanging
            if idx is None:
                self.logger.info("Worker recived end Task, worker exit")
                break
            try:
                self.process_task(model, idx, input_data)
            except Exception as e:
                print(f"Error in worker: {e}, worker exit")
                traceback.print_exc()
                break
        self.worker_loaded_count.value -= 1

    def is_alive(self):
        return any(p.is_alive() for p in self.worker_list)

    def __getstate__(self):
        # 防止 pickle.dump() 时报错, 删除 `processor_list` 属性
        state = self.__dict__.copy()
        if "worker_list" in state:
            del state["worker_list"]
        return state

    def start_workers(self):
        # Start processes.
        for _ in range(self.num_workers):
            p = mp.Process(target=self.worker)
            p.start()
            self.worker_list.append(p)

    def stop_workers(self):
        # Send stop signals to workers (None as a sentinel value).
        for _ in range(self.num_workers):
            self.task_queue.put((None, None))  # Sentinel to stop workers
        # Wait for all processes to finish.
        for p in self.worker_list:
            p.terminate()
            p.join()
        self.worker_list.clear()
        self.worker_loaded_count.value = 0
        return [self.results[key] for key in sorted(self.results.keys())]


class AsyncWorker(InferenceWorker):
    def __init__(self, worker_timeout=60):
        super().__init__(worker_timeout)
        self.task = None

    def start_workers(self):
        async def async_worker():
            return await asyncio.to_thread(self.worker)

        self.task = asyncio.create_task(async_worker())

    def stop_workers(self):
        if self.task is not None:
            self.task.cancel()
            self.task = None

    def is_alive(self):
        # return True
        return self.task is not None and not self.task.done()


class CustomInference(MultiProcessInference):
    def infer_task(self, model, input_data):
        """
        Custom inference task implementation: Accepts multiple input parameters.
        """
        time.sleep(0.2)  # Simulate processing time.
        with torch.no_grad():
            result = model(input_data["input"].cuda())
        # Combine output with metadata
        return {"output": result.cpu().numpy(), "info": input_data["meta"]}

    def get_model(self):
        """
        Load the model for inference. This should be overridden if needed.
        :return: The model to be used in inference.
        """
        # You should load your actual model here.
        return torch.nn.Linear(10, 1).cuda()  # Example model.


async def main():
    mp.set_start_method("spawn", force=True)  # Use 'spawn' start method for CUDA.
    model = torch.nn.Linear(10, 1).cuda()  # Example model moved to CUDA.
    # Generate 100 tasks, each with "input" data and "meta" information.
    tasks = [{"input": torch.randn(1, 10), "meta": f"task-{i}"} for i in range(50)]
    infer = CustomInference(num_workers=2)
    infer.start_workers()
    r1 = infer.add_tasks(tasks)
    r2 = infer.add_tasks(tasks)
    results = await infer.wait_for_results(len(r1) + len(r2))
    infer.stop_workers()
    print("All tasks completed!")
    print("Example result:", results[:5])  # Print a subset of the results.


if __name__ == "__main__":
    asyncio.run(main())
