import pynvml
import time
import sys

def monitor_pcie_throughput(gpu_index=0, interval=1.0):
    try:
        # NVML 초기화
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        device_name = pynvml.nvmlDeviceGetName(handle)
        
        print(f"Monitoring PCIe Throughput for: {device_name} (GPU {gpu_index})")
        print(f"{'Time':<10} | {'RX (MB/s)':<12} | {'TX (MB/s)':<12}")
        print("-" * 40)

        while True:
            # PCIe TX: GPU -> CPU (Transmit)
            tx_usage = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            # PCIe RX: CPU -> GPU (Receive)
            rx_usage = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)

            # KB/s -> MB/s 변환
            tx_mb = tx_usage / 1024.0
            rx_mb = rx_usage / 1024.0

            current_time = time.strftime("%H:%M:%S")
            print(f"{current_time:<10} | {rx_mb:>10.2f}   | {tx_mb:>10.2f}")

            time.sleep(interval)

    except pynvml.NVMLError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    # 첫 번째 GPU(0번) 모니터링
    monitor_pcie_throughput(gpu_index=0)
