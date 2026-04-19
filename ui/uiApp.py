import customtkinter as ctk
import threading
import time
import json
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from datetime import datetime
from terminal_run import run_compression, stop_compression
import platform
from tkinter import filedialog
from terminal_run import run_compression


########################################################################################################################
# Setup

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Distiller Training UI")

app.state('zoomed')


########################################################################################################################
# Setting Global Variables

running = False
loss_values = []
acc_values = []
epochs_list = []


########################################################################################################################
# Layout Frames

# Frame for the entire page
main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Frame for the left side that will contain the input config
left_frame = ctk.CTkFrame(main_frame, width=300)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

# Frame for MatPlotLib
right_frame = ctk.CTkFrame(main_frame)
right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)


########################################################################################################################
# INPUTS (LEFT FRAME)

ctk.CTkLabel(left_frame, text="Training Config", font=("Arial", 16)).pack(pady=10)

# Learning Rate Row
lr_row = ctk.CTkFrame(left_frame)
lr_row.pack(fill="x", pady=5)

ctk.CTkLabel(lr_row, text="Learning Rate").pack(side="left", padx=10)

lr_entry = ctk.CTkEntry(lr_row)
lr_entry.insert(0, "0.001")
lr_entry.pack(side="right", padx=10)

# Epochs Row
epoch_row = ctk.CTkFrame(left_frame)
epoch_row.pack(fill="x", pady=5)

ctk.CTkLabel(epoch_row, text="Epochs").pack(side="left", padx=10)

epoch_entry = ctk.CTkEntry(epoch_row)
epoch_entry.insert(0, "10")
epoch_entry.pack(side="right", padx=10)

# Batch Size Row
batch_row = ctk.CTkFrame(left_frame)
batch_row.pack(fill="x", pady=5)

ctk.CTkLabel(batch_row, text="Batch Size").pack(side="left", padx=10)

batch_entry = ctk.CTkEntry(batch_row)
batch_entry.insert(0, "32")
batch_entry.pack(side="right", padx=10)

# Scheduler Row
scheduler_row = ctk.CTkFrame(left_frame)
scheduler_row.pack(fill="x", pady=5)

ctk.CTkLabel(scheduler_row, text="Scheduler").pack(side="left", padx=10)

scheduler_path = None  # global variable

def select_scheduler():
    global scheduler_path
    file_path = filedialog.askopenfilename(
        title="Select Scheduler YAML",
        filetypes=[("YAML files", "*.yaml *.yml")]
    )
    if file_path:
        scheduler_path = file_path
        scheduler_label.configure(text="Selected")

scheduler_label = ctk.CTkLabel(scheduler_row, text="None")
scheduler_label.pack(side="right", padx=5)

browse_button = ctk.CTkButton(
    scheduler_row,
    text="Browse",
    width=70,
    command=select_scheduler
)
browse_button.pack(side="right", padx=5)

# Model Row
model_row = ctk.CTkFrame(left_frame)
model_row.pack(fill="x", pady=5)

ctk.CTkLabel(model_row, text="Model").pack(side="left", padx=10)

model_dropdown = ctk.CTkOptionMenu(
    model_row,
    values=["resnet20_cifar", "resnet50", "mobilenet_v2", "vgg16"]
)
model_dropdown.set("resnet20_cifar")
model_dropdown.pack(side="right", padx=10)

# STATUS Section
status_label = ctk.CTkLabel(left_frame, text="Waiting to Run", text_color="green")
status_label.pack(pady=10)

progress_bar = ctk.CTkProgressBar(left_frame)
progress_bar.set(0)
progress_bar.pack(fill="x", pady=10)

time_label = ctk.CTkLabel(left_frame, text="")
time_label.pack(pady=5)


########################################################################################################################
# GRAPH (RIGHT TOP)

graph_frame = ctk.CTkFrame(right_frame, height=500)
graph_frame.pack(fill="both", expand=False, pady=(0, 10))
graph_frame.pack_propagate(False)

# Create 2 side-by-side graphs
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy graph (LEFT)
ax_acc.set_title("Accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy (%)")

# Loss graph (RIGHT)
ax_loss.set_title("Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")

canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)


########################################################################################################################
# TERMINAL OUTPUT (RIGHT BOTTOM)

output_box = ctk.CTkTextbox(right_frame, height=180)
output_box.pack(fill="both", expand=True, pady=(0, 10))


########################################################################################################################
# TERMINAL OUTPUT

class TextRedirector:
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        # self.textbox.insert("end", datetime.now())
        self.textbox.insert("end", text)
        self.textbox.see("end")  # Auto-scroll

    def flush(self):
        pass
 
sys.stdout = TextRedirector(output_box)
sys.stderr = TextRedirector(output_box)

########################################################################################################################
# PARSING TERMINAL OUTPUT

def parse_training_line(line):
    line = line.strip()
    if not line.startswith('Epoch:'):
        return None
    epoch_match = re.match(r'^Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)\s*/\s*(\d+)\]\s*(.*)$', line)
    if not epoch_match:
        return None
    epoch = int(epoch_match.group(1))
    rest = epoch_match.group(4)
    stats = {}
    for match in re.finditer(r'([A-Za-z0-9_ ]+?)\s+(-?\d+\.\d+|-?\d+)(?=\s|$)', rest):
        name = match.group(1).strip()
        value_str = match.group(2)
        value = float(value_str) if '.' in value_str else int(value_str)
        stats[name] = value
    return {'epoch': epoch, 'stats': stats}


def update_graph():
    # Clear both graphs
    ax_acc.clear()
    ax_loss.clear()

    # Accuracy graph (LEFT)
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.plot(
        epochs_list,
        acc_values,
        label="Accuracy"
    )

    # Loss graph (RIGHT)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.plot(
        epochs_list,
        loss_values,
        label="Loss"
    )

    canvas.draw()

########################################################################################################################
# TRAINING FUNCTION


def training():
    global running
    model = model_dropdown.get()
    learning_rate = float(lr_entry.get())
    epochs = int(epoch_entry.get())
    batch_size = int(batch_entry.get())

    running = True
    run_button.configure(state="disabled")
    status_label.configure(text="Running...", text_color="yellow")
    output_box.delete("1.0", "end")
    progress_bar.set(0)
    time_label.configure(text="")

    loss_values.clear()
    acc_values.clear()
    epochs_list.clear()
    epoch_plot_data = {}

    def handle_line(line):
        parsed = parse_training_line(line)
        if parsed is None:
            return

        epoch = parsed['epoch']
        stats = parsed['stats']
        loss = stats.get('Overall Loss') or stats.get('Objective Loss') or stats.get('Loss')
        acc = stats.get('Top1') or stats.get('Top1_exit0')

        if loss is None and acc is None:
            return

        epoch_plot_data[epoch] = {
            'loss': loss,
            'acc': acc
        }

        sorted_epochs = sorted(epoch_plot_data)
        epochs_list[:] = [e + 1 for e in sorted_epochs]
        loss_values[:] = [epoch_plot_data[e].get('loss', float('nan')) for e in sorted_epochs]
        acc_values[:] = [epoch_plot_data[e].get('acc', float('nan')) for e in sorted_epochs]

        app.after(0, update_graph)

        if epochs > 0:
            progress = min(1.0, (epoch + 1) / epochs)
            app.after(0, lambda p=progress: progress_bar.set(p))

    def training_finished(success=True):
        status = "Completed" if success else "Failed"
        color = "green" if success else "red"
        status_label.configure(text=status, text_color=color)
        run_button.configure(state="normal")
        if not success:
            output_box.insert("end", "\nCompression command finished with errors.\n")

    try:
        run_compression(model, learning_rate, epochs, batch_size,scheduler_path, line_handler=handle_line)
        app.after(0, lambda: training_finished(success=True))
    except Exception:
        app.after(0, lambda: training_finished(success=False))
    finally:
        running = False


########################################################################################################################
# BUTTON FUNCTIONS

def start_training():
    thread = threading.Thread(target=training)
    thread.start()

def stop_training():
    global running

    running = False
    stop_compression()

    status_label.configure(text="Killing Process, Please Wait...", text_color="red")
    run_button.configure(state="normal")


########################################################################################################################
# DEFINING BUTTON FUNCTIONS REFERENCES

button_frame = ctk.CTkFrame(left_frame)
button_frame.pack()
run_button = ctk.CTkButton(button_frame, text="Run", command=start_training)
run_button.pack(side="left", padx=5)
ctk.CTkButton(button_frame, text="Stop", command=stop_training, fg_color="red").pack(side="left", padx=5)


########################################################################################################################
# START APP

#edit this bit out later
def on_close():
    app.destroy()
    exit()

app.protocol("WM_DELETE_WINDOW", on_close)

app.mainloop()