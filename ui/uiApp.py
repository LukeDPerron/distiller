import customtkinter as ctk
import threading
import time
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


########################################################################################################################
# Setup

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Distiller Training UI")


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

# Model Row
model_row = ctk.CTkFrame(left_frame)
model_row.pack(fill="x", pady=5)

ctk.CTkLabel(model_row, text="Model").pack(side="left", padx=10)

model_dropdown = ctk.CTkOptionMenu(
    model_row,
    values=["resnet20", "resnet50", "mobilenet_v2", "vgg16"]
)
model_dropdown.set("resnet20")
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

graph_frame = ctk.CTkFrame(right_frame)
graph_frame.pack(fill="both", expand=True)

fig, ax = plt.subplots()
ax.set_title("Training Metrics")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")


canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)


########################################################################################################################
# MATPLOTLIB

output_box = ctk.CTkTextbox(right_frame, height=180)
output_box.pack(fill="x", pady=10)


########################################################################################################################
# BACKEND PLACEHOLDER (FOR DISTILLER LATER)

def distiller_step(epoch):
    """
    🔌 Replace this later with real model training call
    """
    loss = 1 / (epoch + 1)
    acc = 80 + epoch
    return loss, acc


########################################################################################################################
# TRAINING LOOP

# AI GENERATED TEST FUNCTION TO SEE IF IT WORKS DURING DEVELOPMENT
def fake_training():
    global running

    running = True

    run_button.configure(state="disabled")
    status_label.configure(text="Running...", text_color="yellow")
    # output_box.delete("1.0", "end")

    epochs = int(epoch_entry.get())
    progress_bar.set(0)

    loss_values.clear()
    acc_values.clear()
    epochs_list.clear()

    for i in range(epochs):
        if not running:
            output_box.insert("end", "\nStopped.\n")
            break

        time.sleep(0.5)

        # backend hook
        loss, acc = distiller_step(i)

        loss_values.append(loss)
        acc_values.append(acc)
        epochs_list.append(i + 1)

        progress = (i + 1) / epochs
        progress_bar.set(progress)

        remaining = (epochs - i) * 0.5
        time_label.configure(text=f"ETA: {remaining:.1f}s")

        output_box.insert(
            "end",
            f"Epoch {i+1}/{epochs} | Loss: {loss:.3f} | Acc: {acc:.1f}%\n"
        )
        output_box.see("end")

        # update graph
        ax.clear()
        ax.plot(epochs_list, loss_values, label="Loss")
        ax.plot(epochs_list, acc_values, label="Accuracy")
        ax.legend()
        canvas.draw()

    status_label.configure(text="Completed", text_color="green")
    run_button.configure(state="normal")
    running = False


########################################################################################################################
# BUTTON FUNCTIONS

def start_training():
    thread = threading.Thread(target=fake_training) # fake_training can be used for testing
    thread.start()

def stop_training():
    global running
    running = False
    status_label.configure(text="Stopped", text_color="red")
    run_button.configure(state="normal")


########################################################################################################################
# DEFINING BUTTON FUNCTIONS REFERENCES

button_frame = ctk.CTkFrame(left_frame)
button_frame.pack()
run_button = ctk.CTkButton(button_frame, text="Run", command=start_training)
run_button.pack_forget()  # remove duplicate placeholder (we already created real one above)

ctk.CTkButton(button_frame, text="Run", command=lambda: start_training()).pack(side="left", padx=5)
ctk.CTkButton(button_frame, text="Stop", command=lambda: stop_training(), fg_color="red").pack(side="left", padx=5)


########################################################################################################################
# START APP

#edit this bit out later
def on_close():
    app.destroy()
    exit()

app.protocol("WM_DELETE_WINDOW", on_close)

app.mainloop()