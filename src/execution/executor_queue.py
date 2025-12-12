import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from src.execution.binance_executor import BinanceExecutor

logger = logging.getLogger("executor_queue")


class ExecutorQueue:
    def __init__(self, executor: BinanceExecutor, max_retries: int = 3):
        self.executor = executor
        self.max_retries = max_retries
        self.order_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        self.recent_errors = []  # Track recent errors

    def get_recent_errors(self) -> list:
        """Get and clear recent errors."""
        errors = self.recent_errors.copy()
        self.recent_errors.clear()
        return errors

    def start(self):
        """Start the worker thread."""
        if self.is_running:
            return
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("🚀 Executor Queue Started")

    def stop(self):
        """Stop the worker thread."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("🛑 Executor Queue Stopped")

    def submit_order(self, order_params: Dict[str, Any]):
        """Submit an order to the queue."""
        self.order_queue.put(order_params)
        logger.info(f"📥 Order Submitted to Queue: {order_params}")

    def _process_queue(self):
        """Worker loop to process orders."""
        while self.is_running:
            try:
                # Get order from queue with timeout to allow checking is_running
                try:
                    order_params = self.order_queue.get(timeout=1)
                except queue.Empty:
                    continue

                self._execute_with_retry(order_params)
                self.order_queue.task_done()

            except Exception as e:
                logger.error(f"Error in executor worker: {e}", exc_info=True)

    def _execute_with_retry(self, order_params: Dict[str, Any]):
        """Execute order with retries and exponential backoff."""
        retries = 0
        backoff = 1  # Start with 1 second

        while retries <= self.max_retries:
            try:
                logger.info(
                    f"⚙️ Executing Order (Attempt {retries+1}/{self.max_retries+1})..."
                )

                # Call the executor
                result = self.executor.execute_order(**order_params)

                if result:
                    logger.info(f"✅ Order Executed Successfully: {result}")
                    return
                else:
                    # If execute_order returns None (e.g. dry run or error handled internally),
                    # we assume it failed or was handled.
                    # If it was a dry run, it logs and returns mock.
                    # If it was an error, execute_order usually logs it.
                    # Let's assume if it returns None/Empty it might be a failure we want to retry if it was an exception.
                    # But BinanceExecutor catches exceptions and returns None.
                    # We might need to modify BinanceExecutor to raise exceptions or return status.
                    # For now, let's assume if it returns None, it failed.
                    logger.warning("⚠️ Order execution returned None.")

            except Exception as e:
                logger.error(f"❌ Execution Failed: {e}")

            # Retry Logic
            retries += 1
            if retries <= self.max_retries:
                logger.info(f"⏳ Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                error_msg = f"Max retries reached. Order Failed: {order_params}"
                logger.error(f"💀 {error_msg}")
                self.recent_errors.append(error_msg)
