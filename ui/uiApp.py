import customtkinter as ctk
import threading
import time

# -------------------------------
# App Setup
# -------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Training UI (Mock)")
app.geometry("750x550")

running = False  # controls fake process

# -------------------------------
# Input Frame
# -------------------------------
input_frame = ctk.CTkFrame(app)
input_frame.pack(pady=10, padx=10, fill="x")

# Learning rate
ctk.CTkLabel(input_frame, text="Learning Rate:").grid(row=0, column=0, padx=10, pady=5)
lr_entry = ctk.CTkEntry(input_frame)
lr_entry.insert(0, "0.001")
lr_entry.grid(row=0, column=1)

# Epochs
ctk.CTkLabel(input_frame, text="Epochs:").grid(row=1, column=0, padx=10, pady=5)
epoch_entry = ctk.CTkEntry(input_frame)
epoch_entry.insert(0, "10")
epoch_entry.grid(row=1, column=1)

# Batch size
ctk.CTkLabel(input_frame, text="Batch Size:").grid(row=2, column=0, padx=10, pady=5)
batch_entry = ctk.CTkEntry(input_frame)
batch_entry.insert(0, "32")
batch_entry.grid(row=2, column=1)

# Model dropdown
ctk.CTkLabel(input_frame, text="Model:").grid(row=3, column=0, padx=10, pady=5)
model_dropdown = ctk.CTkOptionMenu(
    input_frame,
    values=["resnet20", "resnet50", "mobilenet_v2", "vgg16"]
)
model_dropdown.set("resnet20")
model_dropdown.grid(row=3, column=1)

# -------------------------------
# Status Label
# -------------------------------
status_label = ctk.CTkLabel(app, text="Status: Idle", text_color="green")
status_label.pack(pady=5)

# -------------------------------
# Progress Bar
# -------------------------------
progress_bar = ctk.CTkProgressBar(app)
progress_bar.set(0)
progress_bar.pack(padx=10, pady=5, fill="x")

# -------------------------------
# Output Box (Fake Terminal)
# -------------------------------
output_box = ctk.CTkTextbox(app, height=250)
output_box.pack(padx=10, pady=10, fill="both", expand=True)

# -------------------------------
# Fake Training Logic
# -------------------------------
def fake_training():
    global running

    running = True
    status_label.configure(text="Status: Running...", text_color="yellow")
    output_box.delete("1.0", "end")
    progress_bar.set(0)

    total_epochs = int(epoch_entry.get())

    for i in range(total_epochs):
        if not running:
            output_box.insert("end", "\nTraining stopped by user.\n")
            return

        # Simulate work
        time.sleep(0.5)

        # Update progress
        progress = (i + 1) / total_epochs
        progress_bar.set(progress)

        # Fake terminal output
        output_box.insert("end", f"Epoch {i+1}/{total_epochs} - Loss: 0.{i+3} Accuracy: {80+i}%\n")
        output_box.see("end")

    status_label.configure(text="Status: Completed", text_color="green")
    running = False

# -------------------------------
# Thread Wrapper
# -------------------------------
def start_training():
    thread = threading.Thread(target=fake_training)
    thread.start()

# -------------------------------
# Stop Button Logic
# -------------------------------
def stop_training():
    global running
    running = False
    status_label.configure(text="Status: Stopped", text_color="red")

# -------------------------------
# Buttons
# -------------------------------
button_frame = ctk.CTkFrame(app)
button_frame.pack(pady=10)

run_button = ctk.CTkButton(button_frame, text="Run", command=start_training)
run_button.grid(row=0, column=0, padx=10)

stop_button = ctk.CTkButton(button_frame, text="Stop", command=stop_training, fg_color="red")
stop_button.grid(row=0, column=1, padx=10)

# -------------------------------
# Start App
# -------------------------------
app.mainloop()