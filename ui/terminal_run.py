import os
import subprocess
import sys

def run_compression(model, learning_rate, epochs, batch_size, scheduler_path=None, line_handler=None):
    os.chdir('../examples/classifier_compression')

    cmd = [
        sys.executable, 'compress_classifier.py',
        '-a', str(model),
        '--lr', str(learning_rate),
        '-p', '100',
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
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end='')
            if line_handler is not None:
                try:
                    line_handler(line)
                except Exception:
                    pass
            sys.stdout.flush()

        process.wait()

        print("\n" + "="*80)
        if process.returncode == 0:
            print("✓ Compression completed!")
        else:
            print(f"✗ Compression failed with exit code {process.returncode}")

    except Exception as e:
        print(f"Error: {e}")