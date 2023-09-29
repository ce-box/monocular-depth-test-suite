import psutil, time
from pathlib import Path
from datetime import datetime


def performance_metrics(func):
    def wrapper(*args, **kwargs):
        # Measure start time
        start_time = time.time()

        # Get CPU, virtual memory, and disk usage metrics before executing the function
        cpu_start = psutil.cpu_percent()
        virtual_memory_start = psutil.virtual_memory().percent
        disk_usage_start = psutil.disk_usage("/").percent

        # Execute the function
        result = func(*args, **kwargs)

        # Get CPU, virtual memory, and disk usage metrics after executing the function
        cpu_end = psutil.cpu_percent()
        virtual_memory_end = psutil.virtual_memory().percent
        disk_usage_end = psutil.disk_usage("/").percent

        # Calculate execution time
        execution_time = time.time() - start_time

        # Write the performance metrics to a file with a timestamp
        with open(
            f"{Path.cwd()}/logs/{func.__name__}_{datetime.now()}.txt", "a"
        ) as file:
            file.write(f"Performance metrics for '{func.__name__}':\n")
            file.write(f"  - Execution time: {execution_time:.2f} seconds\n")
            file.write(f"  - CPU usage: {cpu_end - cpu_start:.2f}%\n")
            file.write(
                f"  - Virtual memory usage: {virtual_memory_end - virtual_memory_start:.2f}%\n"
            )
            file.write(f"  - Disk usage: {disk_usage_end - disk_usage_start:.2f}%\n\n")

        return result

    return wrapper
