import os
import subprocess
import sys

current_process = None
original_dir = os.getcwd()

def run_compression(model, learning_rate, printing_frequency, epochs, batch_size, scheduler_path=None, line_handler=None):
    os.chdir('../examples/classifier_compression')

    cmd = [
        sys.executable, 'compress_classifier.py',
        '-a', str(model),
        '--lr', str(learning_rate),
        '-p', str(printing_frequency),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        './data.cifar10',
        '-j', '2'
    ]

    if scheduler_path:
        cmd.extend(["--compress", scheduler_path])

    print("Running compression command:")
    print(" ".join(cmd))
    print("\n" + "="*80 + "\n")

    try:
        global current_process

        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in current_process.stdout:
            print(line, end='')
            if line_handler is not None:
                try:
                    line_handler(line)
                except Exception:
                    pass
            sys.stdout.flush()

        current_process.wait()

        print("\n" + "="*80)
        if current_process.returncode == 0:
            print("✓ Compression completed!")
        else:
            print(f"✗ Compression failed with exit code {current_process.returncode}")

    except Exception as e:
        print(f"Error: {e}")


def stop_compression():
    global current_process

    if current_process is not None:
        try:
            current_process.kill()
            print("\nStopping process...\n")
            current_process = None
            os.chdir(original_dir)
        except Exception as e:
            print(f"Error stopping process: {e}")