import numpy as np
import pandas as pd
import tensorflow as tf
import json
import time
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("  NDT Coconut Ripeness — LSTM TinyML Pipeline")
print("=" * 60)

# =============================================================
# STEP 1 — DATA LOADING & PREPROCESSING
# =============================================================

print("\n[1/6] Loading and preprocessing dataset...")

df = pd.read_csv("coconut_training_data.csv")

# Drop pre-strike ambient noise columns (sample_0, sample_1).
# For the LSTM this is especially important — recurrent units
# carry hidden state across ALL timesteps. If the first two
# timesteps contain meaningless noise (10-25 ADC counts), the
# hidden state h_0 and h_1 will be polluted before the LSTM
# ever reaches the informative strike peak at sample_2.
# Dropping them means the LSTM's first meaningful hidden state
# is computed directly from the strike impact.
df = df.drop(columns=["sample_0", "sample_1"])

# Encode labels: Unripe=0, Ripe=1.
df["label"] = df["label"].map({"Unripe": 0, "Ripe": 1})

X = df.drop(columns=["label"]).values   # shape: (3000, 30)
y = df["label"].values                  # shape: (3000,)

# Stratified 80/10/10 split.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# MinMaxScaler fitted on training data only.
# For LSTMs, normalisation is even more critical than for CNNs.
# The tanh activation is saturated by inputs outside [-2, 2].
# Raw ADC values (0-4095) would push every gate activation into
# saturation, making gradients vanish and training fail entirely.
# MinMax scaling to [0,1] keeps all inputs in the tanh linear
# regime, allowing the gates to modulate state meaningfully.
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Save scaler parameters for ESP32 firmware.
scaler_params = {
    "data_min" : scaler.data_min_.tolist(),
    "data_max" : scaler.data_max_.tolist(),
    "scale"    : scaler.scale_.tolist(),
    "min"      : scaler.min_.tolist()
}
with open("scaler_params_lstm.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

# Reshape for LSTM: (batch_size, timesteps, features).
# 30 timesteps = sample_2 through sample_31.
# 1 feature per timestep = single piezo ADC reading.
# The LSTM processes this as a sequence: at each of the 30
# steps it receives one ADC value and updates its hidden state
# h_t and cell state c_t. After step 30, return_sequences=False
# means only the final h_30 is passed to the Dense output layer.
X_train = X_train.reshape(-1, 30, 1)
X_val   = X_val.reshape(-1, 30, 1)
X_test  = X_test.reshape(-1, 30, 1)

print(f"    Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"    LSTM input: 30 timesteps, 1 feature/step (univariate)")
print(f"    Label balance — Train Ripe: {y_train.sum()}  Unripe: {(y_train==0).sum()}")

# =============================================================
# STEP 2 — MODEL ARCHITECTURE
# =============================================================

print("\n[2/6] Building LSTM architecture...")

# WHY ONLY 8 LSTM UNITS:
# An LSTM unit with n_units has 4 gate matrices, each of size:
#   (input_size + n_units) x n_units
# Total LSTM parameters = 4 * (input_features + n_units) * n_units
#                                   + 4 * n_units (bias)
# For 8 units, 1 input feature:
#   4 * (1 + 8) * 8 + 4 * 8 = 288 + 32 = 320 parameters
# For 32 units (more typical):
#   4 * (1 + 32) * 32 + 4 * 32 = 4224 + 128 = 4352 parameters
# 32-unit LSTM = 13x more parameters for marginal accuracy gain.
# On ESP32-C3 with 400KB SRAM, the LSTM hidden state, cell state,
# and all 4 gate activations must live in RAM simultaneously.
# 8 units is the smallest practical configuration that can still
# learn the exponential decay rate difference between classes.

model = tf.keras.Sequential([

    # LSTM layer: 8 units.
    # activation='tanh'        : output gate and cell state squash.
    # recurrent_activation='sigmoid': forget/input/output gates.
    # return_sequences=False   : only return final hidden state h_30.
    #                           We do not need all 30 intermediate
    #                           states — only the final summary of
    #                           the entire decay sequence.
    # unroll=True              : unrolls the RNN loop at graph build
    #                           time. Faster inference on small fixed
    #                           sequences like ours (30 steps).
    #                           Avoids dynamic loop overhead on ESP32.
    tf.keras.layers.LSTM(
        units=8,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=False,
        unroll=True,
        input_shape=(30, 1),
        name="lstm_decay_encoder"
    ),

    # Output: single sigmoid neuron.
    # The 8-dimensional final hidden state h_30 encodes a summary
    # of the entire 30-step waveform. The Dense layer learns a
    # linear combination of these 8 state values that best
    # separates Ripe (slow decay, h_30 still active) from
    # Unripe (fast decay, h_30 near zero by step 5).
    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )
], name="coconut_lstm")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

total_params = model.count_params()
print(f"\n    Total trainable parameters : {total_params:,}")
print(f"    LSTM parameter breakdown   :")
print(f"      4 gates x (1 input + 8 units) x 8 units = {4*(1+8)*8} weight params")
print(f"      4 gates x 8 bias terms                  = {4*8} bias params")
print(f"      Dense: 8 x 1 + 1 bias                   = {8*1+1} params")

# =============================================================
# STEP 3 — TRAINING
# =============================================================

print("\n[3/6] Training model...")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(f"\n    Training stopped at epoch : {len(history.history['loss'])}")
print(f"    Best val_loss             : {min(history.history['val_loss']):.4f}")
print(f"    Best val_accuracy         : {max(history.history['val_accuracy']):.4f}")

# =============================================================
# STEP 4 — TFLITE CONVERSION & QUANTIZATION
# =============================================================

print("\n[4/6] Converting to TFLite...")
print("    NOTE: LSTM quantization compatibility varies by TF version.")
print("    Attempting standard INT8 builtins first...")

TFLITE_PATH = "coconut_lstm.tflite"
quantization_mode = "unknown"

def representative_dataset_generator():
    # Calibration data for INT8 quantization.
    # Shape must match LSTM input: (1, 30, 1).
    calibration_data = X_train[:200].astype(np.float32)
    for sample in calibration_data:
        yield [np.expand_dims(sample, axis=0)]

# ── Attempt 1: Full INT8 standard builtins (best for ESP32) ──
# This is the ideal path — all ops map to TFLite Micro built-ins
# which are available on the ESP32 without any extra op resolver.
# LSTM ops are supported as TFLITE_BUILTINS_INT8 in TF >= 2.8.
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model     = converter.convert()
    quantization_mode = "INT8 standard builtins only"
    print("    SUCCESS: Standard INT8 quantization completed.")

except Exception as e1:
    print(f"    INT8-only failed: {e1}")
    print("    Falling back to TFLITE_BUILTINS + SELECT_TF_OPS...")

    # ── Attempt 2: Builtins + Select TF Ops fallback ──
    # Some LSTM variants (especially with certain recurrent
    # activation functions) require Select TF Ops in older TF
    # versions. This path works on ESP32 but requires adding
    # the SelectiveRegistration op resolver in firmware:
    # #include "tensorflow/lite/micro/all_ops_resolver.h"
    # instead of MicroMutableOpResolver with specific ops only.
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_generator
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model     = converter.convert()
        quantization_mode = "Mixed: TFLITE_BUILTINS + SELECT_TF_OPS (fallback)"
        print("    SUCCESS: Fallback with SELECT_TF_OPS completed.")
        print("    WARNING: ESP32 firmware must use AllOpsResolver.")

    except Exception as e2:
        print(f"    SELECT_TF_OPS also failed: {e2}")
        print("    Final fallback: float32 dynamic-range quantization...")

        # ── Attempt 3: Dynamic range quantization (float32 ops) ──
        # Weights are quantized to INT8 for smaller Flash footprint
        # but activations remain float32 at runtime. Works on all
        # TF versions. Larger RAM than full INT8, but fully
        # compatible. Inference on ESP32 will be slower.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model     = converter.convert()
        quantization_mode = "Dynamic range (float32 activations — final fallback)"
        print("    Applied dynamic range quantization (float32 activations).")

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

flash_kb = os.path.getsize(TFLITE_PATH) / 1024.0
print(f"\n    TFLite model saved      : {TFLITE_PATH}")
print(f"    Quantization mode used  : {quantization_mode}")
print(f"    Flash footprint         : {flash_kb:.2f} KB")

# =============================================================
# STEP 5 — METRIC EVALUATION
# =============================================================

print("\n[5/6] Evaluating all 9 benchmark metrics...")

# ── ML Metrics (Keras float32 for reliable probabilities) ──

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

accuracy  = accuracy_score(y_test, y_pred)  * 100
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred,    zero_division=0)
f1        = f1_score(y_test, y_pred,        zero_division=0)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n    Accuracy  : {accuracy:.2f}%")
print(f"    Precision : {precision:.4f}")
print(f"    Recall    : {recall:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print(f"    Confusion Matrix:\n{cm}")

# ── Confusion Matrix Plot ──

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Unripe (0)", "Ripe (1)"],
    yticklabels=["Unripe (0)", "Ripe (1)"],
    ax=ax
)
ax.set_title("Confusion Matrix — LSTM (Test Set)", fontsize=13, pad=12)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label",      fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix_lstm.png", dpi=150)
plt.close()
print("\n    Confusion matrix saved: confusion_matrix_lstm.png")

# ── Hardware Metric 2: Peak RAM Arena Estimate ──
#
# LSTM RAM is fundamentally different from CNN RAM.
# CNNs need RAM for activation maps (spatially local).
# LSTMs need RAM for ALL of the following simultaneously:
#
# (A) INPUT BUFFER per timestep:
#     (1, 1) float32 = 4 bytes (one ADC reading at a time)
#
# (B) HIDDEN STATE h_t:
#     (1, 8) float32 = 8 * 4 = 32 bytes
#     Must persist across ALL 30 timesteps in RAM.
#
# (C) CELL STATE c_t:
#     (1, 8) float32 = 8 * 4 = 32 bytes
#     Must persist across ALL 30 timesteps in RAM.
#
# (D) GATE ACTIVATIONS (all 4 gates computed simultaneously):
#     Forget gate f_t   : (1, 8) = 32 bytes
#     Input gate i_t    : (1, 8) = 32 bytes
#     Cell gate g_t     : (1, 8) = 32 bytes
#     Output gate o_t   : (1, 8) = 32 bytes
#     Total gates       : 128 bytes
#
# (E) WEIGHT MATRICES (in RAM during inference, not Flash):
#     Kernel W  : (1, 32) float32  = 1*32*4  = 128 bytes
#     Recurrent : (8, 32) float32  = 8*32*4  = 1024 bytes
#     Bias      : (32,)   float32  = 32*4    = 128 bytes
#     Total weights in RAM         : 1280 bytes
#
# (F) OUTPUT Dense layer:
#     Weights: (8,1) = 32 bytes | Bias: (1,) = 4 bytes = 36 bytes
#
# Total raw = A + B + C + D + E + F
# +30% TFLite Micro overhead for scratch buffers and alignment

input_buf    = 1  * 4          # single timestep input
h_state_buf  = 8  * 4          # hidden state
c_state_buf  = 8  * 4          # cell state
gates_buf    = 4 * 8 * 4       # all 4 gate activations
kernel_buf   = 1  * 32 * 4     # input kernel W
recurrent_buf= 8  * 32 * 4     # recurrent kernel U
bias_buf     = 32 * 4          # LSTM bias
dense_buf    = (8 * 1 + 1) * 4 # Dense weights + bias

raw_bytes  = (input_buf + h_state_buf + c_state_buf + gates_buf +
              kernel_buf + recurrent_buf + bias_buf + dense_buf)
overhead   = 1.30
arena_bytes = int(raw_bytes * overhead)
arena_kb    = arena_bytes / 1024.0

print(f"\n    LSTM Peak RAM estimate:")
print(f"      Hidden state h_t  (8 units)  : {h_state_buf}  bytes")
print(f"      Cell state   c_t  (8 units)  : {c_state_buf}  bytes")
print(f"      4 gate activations           : {gates_buf} bytes")
print(f"      Input kernel W    (1x32)     : {kernel_buf} bytes")
print(f"      Recurrent kernel U (8x32)    : {recurrent_buf} bytes")
print(f"      LSTM bias         (32,)      : {bias_buf} bytes")
print(f"      Dense layer                  : {dense_buf}  bytes")
print(f"      Raw total                    : {raw_bytes} bytes")
print(f"      +30%% overhead               : {arena_bytes} bytes")
print(f"      Estimated peak RAM arena     : {arena_kb:.2f} KB")
print(f"    NOTE: h_t and c_t must persist for all 30 timesteps.")
print(f"    This is the fundamental RAM cost of recurrent models.")

# ── Hardware Metric 3: Inference Latency (TFLite, PC baseline) ──
# LSTMs are inherently sequential — each timestep depends on the
# previous hidden state. Unlike CNNs which parallelise across
# the spatial dimension, an LSTM MUST process all 30 timesteps
# one-by-one. This makes LSTM latency significantly higher than
# 1D-CNN for identical parameter counts on the ESP32.

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype      = input_details[0]["dtype"]
input_scale      = input_details[0]["quantization"][0]
input_zero_point = input_details[0]["quantization"][1]

latencies = []

for i in range(len(X_test)):
    single_sample = X_test[i].astype(np.float32)   # shape: (30, 1)

    # Quantize to INT8 if model was successfully INT8 quantized.
    # If dynamic range fallback was used, input stays float32.
    if input_dtype == np.int8 and input_scale != 0:
        sample_q = (single_sample / input_scale + input_zero_point)
        sample_q = np.clip(sample_q, -128, 127).astype(np.int8)
    elif input_dtype == np.int8:
        sample_q = single_sample.astype(np.int8)
    else:
        sample_q = single_sample  # float32 fallback path

    # Add batch dimension: (30,1) → (1,30,1)
    input_data = np.expand_dims(sample_q, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)

    t_start = time.perf_counter()
    interpreter.invoke()
    t_end   = time.perf_counter()

    latencies.append((t_end - t_start) * 1000.0)

avg_latency_ms = np.mean(latencies)
std_latency_ms = np.std(latencies)

print(f"\n    Avg inference latency (PC baseline) : {avg_latency_ms:.4f} ms")
print(f"    Std deviation                        : {std_latency_ms:.4f} ms")
print(f"    ESP32-C3 @ 160MHz estimated          : {avg_latency_ms*25:.1f} – {avg_latency_ms*40:.1f} ms")
print(f"    Expected: higher than 1D-CNN due to sequential timestep processing.")

# =============================================================
# STEP 6 — SAVE BENCHMARK RESULTS TO JSON
# =============================================================

print("\n[6/6] Saving benchmark results to JSON...")

benchmark_results = {
    "model": "LSTM (TinyML — ESP32 Optimised)",
    "dataset": {
        "total_samples"    : len(df),
        "features_used"    : "sample_2 to sample_31 (30 timesteps)",
        "features_dropped" : "sample_0, sample_1 (pre-strike noise)",
        "train_samples"    : int(len(X_train)),
        "val_samples"      : int(len(X_val)),
        "test_samples"     : int(len(X_test))
    },
    "architecture": {
        "input_shape"      : [30, 1],
        "layers"           : [
            "LSTM(8 units, tanh, recurrent=sigmoid, return_sequences=False, unroll=True)",
            "Dense(1, sigmoid)"
        ],
        "total_parameters" : int(total_params),
        "lstm_param_breakdown": {
            "gate_weights" : int(4 * (1 + 8) * 8),
            "gate_biases"  : int(4 * 8),
            "dense_params" : int(8 * 1 + 1)
        },
        "optimizer"        : "Adam (lr=0.001)",
        "loss"             : "binary_crossentropy",
        "early_stopping"   : "patience=5, monitor=val_loss",
        "design_note"      : (
            "8 LSTM units chosen to minimise RAM. Each additional unit "
            "adds (1+n)*4 + n*4 parameters across all gates AND requires "
            "persistent h_t and c_t state tensors in RAM for 30 timesteps."
        )
    },
    "training": {
        "epochs_trained"   : len(history.history["loss"]),
        "best_val_loss"    : float(f"{min(history.history['val_loss']):.4f}"),
        "best_val_accuracy": float(f"{max(history.history['val_accuracy']):.4f}")
    },
    "ml_metrics": {
        "1_accuracy_pct"   : float(f"{accuracy:.2f}"),
        "2_precision"      : float(f"{precision:.4f}"),
        "3_recall"         : float(f"{recall:.4f}"),
        "4_f1_score"       : float(f"{f1:.4f}"),
        "5_confusion_matrix": {
            "true_negative"  : int(cm[0][0]),
            "false_positive" : int(cm[0][1]),
            "false_negative" : int(cm[1][0]),
            "true_positive"  : int(cm[1][1]),
            "saved_as"       : "confusion_matrix_lstm.png"
        }
    },
    "hardware_metrics": {
        "6_flash_memory_kb": float(f"{flash_kb:.2f}"),
        "7_peak_ram_kb"    : float(f"{arena_kb:.2f}"),
        "7_ram_breakdown"  : {
            "hidden_state_h_t_bytes"   : h_state_buf,
            "cell_state_c_t_bytes"     : c_state_buf,
            "gate_activations_bytes"   : gates_buf,
            "input_kernel_bytes"       : kernel_buf,
            "recurrent_kernel_bytes"   : recurrent_buf,
            "lstm_bias_bytes"          : bias_buf,
            "dense_layer_bytes"        : dense_buf,
            "raw_total_bytes"          : raw_bytes,
            "overhead_factor"          : "30%",
            "total_arena_bytes"        : arena_bytes,
            "key_note"                 : (
                "h_t and c_t must persist across all 30 timesteps. "
                "This is unavoidable in recurrent architectures and "
                "is why LSTMs have higher RAM than CNNs at equal "
                "parameter counts."
            )
        },
        "8_inference_latency_ms": {
            "pc_baseline_avg"       : float(f"{avg_latency_ms:.4f}"),
            "pc_baseline_std"       : float(f"{std_latency_ms:.4f}"),
            "esp32_c3_estimated_ms" : f"{avg_latency_ms*25:.1f} – {avg_latency_ms*40:.1f}",
            "note"                  : (
                "LSTM processes 30 timesteps sequentially — no parallelism. "
                "Expected higher latency than 1D-CNN at equivalent parameter count."
            )
        },
        "9_power_consumption": "N/A - Requires Physical Multimeter"
    },
    "tflite_model": {
        "path"              : TFLITE_PATH,
        "quantization_used" : quantization_mode,
        "input_dtype"       : str(input_dtype),
        "esp32_note"        : (
            "If SELECT_TF_OPS was used, ESP32 firmware must use "
            "AllOpsResolver instead of MicroMutableOpResolver. "
            "Standard INT8 builtins preferred for microcontroller deployment."
        )
    },
    "scaler": {
        "type"         : "MinMaxScaler",
        "fitted_on"    : "training set only (no leakage)",
        "params_saved" : "scaler_params_lstm.json",
        "note"         : (
            "Normalisation to [0,1] is critical for LSTM. "
            "Raw ADC values (0-4095) saturate tanh gates, "
            "causing vanishing gradients and training failure."
        )
    }
}

JSON_PATH = "benchmark_lstm_results.json"
with open(JSON_PATH, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print(f"    Benchmark results saved: {JSON_PATH}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — All outputs generated successfully")
print("=" * 60)
print(f"  coconut_lstm.tflite           → Flash to ESP32")
print(f"  benchmark_lstm_results.json   → Paper metrics")
print(f"  confusion_matrix_lstm.png     → Paper figure")
print(f"  scaler_params_lstm.json       → Hardcode into firmware")
print("=" * 60)
print(f"\n  QUICK METRICS SUMMARY")
print(f"  Accuracy         : {accuracy:.2f}%")
print(f"  F1-Score         : {f1:.4f}")
print(f"  Flash (KB)       : {flash_kb:.2f}")
print(f"  RAM arena (KB)   : {arena_kb:.2f}")
print(f"  Latency PC (ms)  : {avg_latency_ms:.4f}")
print(f"  Quantization     : {quantization_mode}")
print("=" * 60)