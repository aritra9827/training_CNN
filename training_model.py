import os
import sys
import time
from datetime import datetime

# Select GPU before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger


# ---------------------------------------------------------------------
# Custom loss function
# ---------------------------------------------------------------------

module_dir = ""
sys.path.append(module_dir)

#from train_module import loss_fro_norm_0_1



#%%
'''loss functions define'''



def loss_fro_norm_0_1(y_true,y_pred):

    '''
    parameters:
    -----------

    ytrue/ypred : numpy ndarray



    returns:
    ---------

    loss    :   real number float

        calculates the frobenius norm of the density matrix computed from the coefficient vectors y_true and y_pred


    '''

    SSize = int(tf.shape(y_true)[1])
#    print('size', SSize, 'print datatype ', type(SSize), 'shape',np.shape(y_true))

    ssize_comp = int(SSize/2)



    y_pred = tf.math.l2_normalize(y_pred, axis = -1)
    y_true = tf.math.l2_normalize(y_true, axis = -1)



    y_true_comp = tf.complex(tf.slice(y_true,begin=[0,0],size=[1,ssize_comp]),
               tf.slice(y_true,begin=[0,ssize_comp],size=[1,ssize_comp]))


    y_pred_comp = tf.complex(tf.slice(y_pred,begin=[0,0],size=[1,ssize_comp]),
               tf.slice(y_pred,begin=[0,ssize_comp],size=[1,ssize_comp]))


    y_true_comp = tf.transpose(y_true_comp); y_pred_comp= tf.transpose(y_pred_comp)




    rho_true = tf.matmul(y_true_comp,tf.linalg.adjoint(y_true_comp))
    rho_pred = tf.matmul(y_pred_comp,tf.linalg.adjoint(y_pred_comp))

    rho_pred = tf.cast(rho_pred,tf.complex128)
    rho_true = tf.cast(rho_true,tf.complex128)


    loss = tf.linalg.norm(rho_true - rho_pred)

    loss_norm = tf.cast(loss,dtype=tf.float32)/tf.cast(tf.sqrt(tf.constant(2.0)),dtype=tf.float32)


    return loss_norm



def loss_fro_norm_v2(y_true, y_pred):



    n = tf.shape(y_true)[-1] // 2

    # Normalize each sample separately
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)

    # Construct complex coefficient vectors
    # Shape: (batch_size, N)
    y_true_comp = tf.complex(
        y_true[:, :n],
        y_true[:, n:]
    )

    y_pred_comp = tf.complex(
        y_pred[:, :n],
        y_pred[:, n:]
    )

    # Convert to column vectors
    # Shape: (batch_size, N, 1)
    y_true_comp = tf.expand_dims(y_true_comp, axis=-1)
    y_pred_comp = tf.expand_dims(y_pred_comp, axis=-1)

    # Density matrices:
    # rho = |psi><psi|
    # Shape: (batch_size, N, N)
    rho_true = tf.matmul(
        y_true_comp,
        tf.linalg.adjoint(y_true_comp)
    )

    rho_pred = tf.matmul(
        y_pred_comp,
        tf.linalg.adjoint(y_pred_comp)
    )

    # Difference between density matrices
    rho_diff = rho_true - rho_pred

    # Frobenius norm for EACH sample:
    # ||A||_F = sqrt(sum_mn |A_mn|^2)
    loss_per_sample = tf.sqrt(
        tf.reduce_sum(
            tf.abs(rho_diff)**2,
            axis=[1, 2]
        )
    )

    # Normalize to [0,1]
    loss_per_sample = loss_per_sample / tf.sqrt(2.0)

    # Mean over the batch
    loss = tf.reduce_mean(loss_per_sample)

    return loss






#%%

# ---------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------

N_x = 10
noise_strength = 0.1

epochs = 9000          # Set this to the actual number used
bsize = 128
learning_rate = 1e-4

architecture_name = "train_900k"

print(datetime.today().strftime("%Y-%m-%d"))
print(f"Noise strength: {noise_strength}")
print(f"Learning rate: {learning_rate}")


# ---------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        print(error)


# ---------------------------------------------------------------------
# Load training and validation data
# ---------------------------------------------------------------------

current_dir = os.getcwd()

train_files = [
    name for name in os.listdir(current_dir)
    if name.endswith("train.h5")
]

val_files = [
    name for name in os.listdir(current_dir)
    if name.endswith("val.h5")
]

if len(train_files) != 1 or len(val_files) != 1:
    raise RuntimeError(
        "Expected exactly one training file and one validation file."
    )

with h5py.File(train_files[0], "r") as hf:
    train_xdata = np.asarray(hf["x_vec"], dtype=np.float32)
    train_ydata = np.asarray(hf["y_vec"], dtype=np.float32)

with h5py.File(val_files[0], "r") as hf:
    val_xdata = np.asarray(hf["x_vec"], dtype=np.float32)
    val_ydata = np.asarray(hf["y_vec"], dtype=np.float32)

print("Training input shape:", train_xdata.shape)
print("Training output shape:", train_ydata.shape)
print("Validation input shape:", val_xdata.shape)
print("Validation output shape:", val_ydata.shape)


# ---------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------

def define_model():

    model = Sequential([
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            input_shape=(
                train_xdata.shape[1],
                train_xdata.shape[2],
                1,
            ),
        ),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),

        Flatten(),
        Dense(128, activation="relu"),
        Dense(train_ydata.shape[1], activation="linear"),
    ])

    return model


model = define_model()

optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss=loss_fro_norm_0_1,
)

model.summary()


# ---------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------

timestamp = str(int(time.time() * 1e6))

model_version = (
    f"train_N{N_x}_epoch{epochs}_noise{noise_strength}_"
    f"{architecture_name}_loss_onsite_largev2"
)

run_info = (
    f"batch{bsize}_lrate{learning_rate}_"
    f"train{train_xdata.shape[0]}"
)

csv_logger = CSVLogger(
    f"{timestamp}_{model_version}_{run_info}.log"
)


# ---------------------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------------------

class SaveModelAndLog(tf.keras.callbacks.Callback):

    def __init__(self, model_dir, log_dir, save_epochs):
        super().__init__()

        self.model_dir = model_dir
        self.log_dir = log_dir
        self.save_epochs = set(save_epochs)

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.history = {}

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        epoch_number = epoch + 1

        self.history.setdefault("epoch", []).append(epoch_number)

        for key, value in logs.items():
            self.history.setdefault(key, []).append(
                None if value is None else np.float32(value)
            )

        if epoch_number not in self.save_epochs:
            return

        model_file = os.path.join(
            self.model_dir,
            f"{timestamp}_{model_version}_{run_info}_"
            f"checkpoint_epoch_{epoch_number}.h5",
        )

        self.model.save(model_file)

        log_file = os.path.join(
            self.log_dir,
            f"{timestamp}_{model_version}_{run_info}_"
            f"checkpoint_epoch_{epoch_number}.log",
        )

        keys = list(self.history.keys())

        with open(log_file, "w") as f:
            f.write(",".join(keys) + "\n")

            for i in range(len(self.history["epoch"])):
                row = []

                for key in keys:
                    value = self.history[key][i]
                    row.append("" if value is None else str(value))

                f.write(",".join(row) + "\n")

        print(f"Model saved at epoch {epoch_number}: {model_file}")


model_dir = (
    "..."
)

log_dir = (
    "..."
)

save_epochs = [2000, 5000, 7500, 8000, 8500, 9000]

checkpoint_callback = SaveModelAndLog(
    model_dir=model_dir,
    log_dir=log_dir,
    save_epochs=save_epochs,
)


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------

start_time = time.time()

history = model.fit(
    train_xdata,
    train_ydata,
    epochs=epochs,
    batch_size=bsize,
    validation_data=(val_xdata, val_ydata),
    verbose=2,
    callbacks=[csv_logger, checkpoint_callback],
)

end_time = time.time()

print(
    f"Training time: {(end_time - start_time) / 3600:.2f} hours"
)


# ---------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------

model.save(
    f"{timestamp}_{model_version}_{run_info}_end.h5"
)


# ---------------------------------------------------------------------
# Plot training history
# ---------------------------------------------------------------------

epoch_array = np.asarray(history.epoch)
loss = np.asarray(history.history["loss"])
val_loss = np.asarray(history.history["val_loss"])

start = 100

plt.figure()
plt.plot(epoch_array[start:], loss[start:], "k", label="Training loss")
plt.plot(epoch_array[start:], val_loss[start:], "r", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"loss_epochs_{timestamp}_{model_version}_{run_info}.png"
)
plt.close()

plt.figure()
plt.plot(
    np.log10(epoch_array[start:]),
    np.log10(loss[start:]),
    "k",
    label="Training loss",
)
plt.plot(
    np.log10(epoch_array[start:]),
    np.log10(val_loss[start:]),
    "r",
    label="Validation loss",
)
plt.xlabel(r"$\log_{10}(\mathrm{epoch})$")
plt.ylabel(r"$\log_{10}(\mathrm{loss})$")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"loss_logscale_{timestamp}_{model_version}_{run_info}.png"
)
plt.close()
