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
print("  NDT Coconut Ripeness — 2D-CNN TinyML Pipeline")
print("=" * 60)

# =============================================================
# STEP 1 — DATA LOADING & PREPROCESSING
# =============================================================

print("\n[1/6] Loading and preprocessing dataset...")

df = pd.read_csv("coconut_training_data.csv")

# Drop pre-strike ambient noise columns (sample_0, sample_1).
# These 2 samples carry only random ADC noise (values 10-25)
# with zero class-discriminating signal. Including them would
# add a meaningless row to our 2D spatial matrix and dilute
# the conv filters that should focus on the decay region.
df = df.drop(columns=["sample_0", "sample_1"])

# Encode labels: Unripe=0, Ripe=1.
df["label"] = df["label"].map({"Unripe": 0, "Ripe": 1})

X = df.drop(columns=["label"]).values   # shape: (3000, 30)
y = df["label"].values                  # shape: (3000,)

# Stratified split: 80% train, 10% val, 10% test.
# Stratify preserves the 50/50 Ripe/Unripe balance in every split.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# MinMaxScaler: fit ONLY on training data.
# Fitting on the full dataset leaks test-set statistics into
# training — a critical error in any published benchmark study.
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Save scaler parameters for ESP32 firmware preprocessing.
# The microcontroller must apply identical normalisation before
# feeding any real ADC reading into the TFLite interpreter.
scaler_params = {
    "data_min" : scaler.data_min_.tolist(),
    "data_max" : scaler.data_max_.tolist(),
    "scale"    : scaler.scale_.tolist(),
    "min"      : scaler.min_.tolist()
}
with open("scaler_params_2dcnn.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

# ── THE CRITICAL 2D RESHAPE ──
# We convert the flat 30-element 1D time-series into a 2D
# spatial matrix of shape (6, 5, 1):
#   6 rows  = 6 temporal windows (each window covers 5 timesteps)
#   5 cols  = 5 consecutive ADC samples within each window
#   1       = single channel (univariate piezo sensor)
#
# Physical interpretation of the 6 windows:
#   Row 0 → sample_2  to sample_6  : Strike peak + immediate drop
#   Row 1 → sample_7  to sample_11 : Early decay region
#   Row 2 → sample_12 to sample_16 : Mid decay region
#   Row 3 → sample_17 to sample_21 : Late decay (Ripe still active)
#   Row 4 → sample_22 to sample_26 : Tail region
#   Row 5 → sample_27 to sample_31 : Near-noise-floor region
#
# IMPORTANT RESEARCH NOTE (acknowledged in paper):
# This reshape imposes spatial neighbourhood relationships that
# do not exist in the original 1D signal. Samples at the END
# of row N and the START of row N+1 are physically adjacent in
# time but spatially non-adjacent in the 2D matrix. A (3,3)
# Conv2D kernel will NOT capture this cross-row temporal
# continuity. The 2D-CNN is therefore learning spatial texture
# patterns in the reshaped matrix — NOT true temporal dynamics.
# This is why the 1D-CNN is the more physically motivated model.
X_train = X_train.reshape(-1, 6, 5, 1)
X_val   = X_val.reshape(-1, 6, 5, 1)
X_test  = X_test.reshape(-1, 6, 5, 1)

print(f"    Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"    2D matrix per sample: (6 temporal windows) x (5 samples/window) x (1 channel)")
print(f"    Label balance — Train Ripe: {y_train.sum()}  Unripe: {(y_train==0).sum()}")

# =============================================================
# STEP 2 — MODEL ARCHITECTURE
# =============================================================

print("\n[2/6] Building 2D-CNN architecture...")

model = tf.keras.Sequential([

    # Conv2D Layer 1: 8 filters, (3,3) kernel, padding='same'.
    # Input: (6, 5, 1) — a 6x5 matrix with 1 channel.
    # padding='same' preserves spatial dimensions → output: (6, 5, 8).
    # A (3,3) kernel sees 3 temporal windows x 3 adjacent samples
    # simultaneously, learning local 2D texture features within
    # the reshaped acoustic matrix (e.g., steep gradient blocks
    # that map to the fast Unripe decay region).
    tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=(6, 5, 1),
        name="conv2d_local_texture"
    ),

    # Conv2D Layer 2: 16 filters, (2,2) kernel, padding='valid'.
    # padding='valid' = no padding, so spatial dims shrink.
    # Input: (6, 5, 8) → Output: (5, 4, 16).
    # This layer detects higher-level 2D patterns — combinations
    # of local textures that differentiate the Ripe "gradual fade"
    # block pattern from the Unripe "sharp cliff then flat" pattern.
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(2, 2),
        activation="relu",
        padding="valid",
        name="conv2d_global_pattern"
    ),

    # GlobalAveragePooling2D collapses (5, 4, 16) → (16,).
    # Alternative Flatten would give 5*4*16 = 320 values → Dense.
    # GAP gives only 16 values → Dense, saving 304 * 16 weights
    # = 4,864 fewer parameters in the Dense layer.
    # On an ESP32-C3 with 400KB SRAM, every KB matters.
    tf.keras.layers.GlobalAveragePooling2D(
        name="global_avg_pool_2d"
    ),

    # Dense 16: learns non-linear combinations of the 16 pooled
    # spatial feature maps into class-discriminating activations.
    tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_classifier"
    ),

    # Output: single sigmoid neuron.
    # Threshold at 0.5 → Ripe (1) or Unripe (0).
    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )
], name="coconut_2dcnn")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

total_params = model.count_params()
print(f"\n    Total trainable parameters: {total_params:,}")

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
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(f"\n    Training stopped at epoch : {len(history.history['loss'])}")
print(f"    Best val_loss             : {min(history.history['val_loss']):.4f}")
print(f"    Best val_accuracy         : {max(history.history['val_accuracy']):.4f}")

# =============================================================
# STEP 4 — TFLITE CONVERSION & INT8 QUANTIZATION
# =============================================================

print("\n[4/6] Converting to TFLite with INT8 post-training quantization...")

# Representative dataset generator for INT8 calibration.
# The quantizer samples these inputs to determine the dynamic
# range of every tensor (weights AND activations), then computes
# the INT8 scale and zero-point for each layer.
# Input shape must match model input: (1, 6, 5, 1).
def representative_dataset_generator():
    calibration_data = X_train[:200].astype(np.float32)
    for sample in calibration_data:
        # Add batch dimension: (6,5,1) → (1,6,5,1)
        yield [np.expand_dims(sample, axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# INT8 quantization: both weights and activations become int8.
# Weights: float32 (4 bytes) → int8 (1 byte) = 4x size reduction.
# Activations: quantized at runtime using calibrated scale/offset.
# This maximises inference speed on ESP32 which lacks an FPU.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

TFLITE_PATH = "coconut_2dcnn.tflite"
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

flash_kb = os.path.getsize(TFLITE_PATH) / 1024.0
print(f"    TFLite model saved : {TFLITE_PATH}")
print(f"    Flash footprint    : {flash_kb:.2f} KB")

# =============================================================
# STEP 5 — METRIC EVALUATION
# =============================================================

print("\n[5/6] Evaluating all 9 benchmark metrics...")

# ── ML Metrics (Keras float32 model for clean probabilities) ──

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
    cmap="Oranges",
    xticklabels=["Unripe (0)", "Ripe (1)"],
    yticklabels=["Unripe (0)", "Ripe (1)"],
    ax=ax
)
ax.set_title("Confusion Matrix — 2D-CNN (Test Set)", fontsize=13, pad=12)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label",      fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix_2dcnn.png", dpi=150)
plt.close()
print("\n    Confusion matrix saved: confusion_matrix_2dcnn.png")

# ── Hardware Metric 2: Peak RAM Arena Estimate ──
# Activation buffers that must exist simultaneously in RAM:
#
# Input tensor     : (1, 6, 5, 1)  = 30  values * 4 bytes = 120  bytes
# After Conv2D-1   : (1, 6, 5, 8)  = 240 values * 4 bytes = 960  bytes
# After Conv2D-2   : (1, 5, 4, 16) = 320 values * 4 bytes = 1280 bytes
# After GAP2D      : (1, 16)       = 16  values * 4 bytes = 64   bytes
# After Dense-16   : (1, 16)       = 16  values * 4 bytes = 64   bytes
# After Output     : (1, 1)        = 1   value  * 4 bytes = 4    bytes
#
# Peak simultaneous usage = input + largest activation (Conv2D-2).
# TFLite Micro allocates the arena to fit the worst-case frame.
# We sum all buffers then add 30% overhead for TFLite Micro
# internal scratch buffers, operator context structs, and
# alignment padding between tensors.

input_buf  = 6 * 5 * 1  * 4    # 120  bytes
conv1_buf  = 6 * 5 * 8  * 4    # 960  bytes
conv2_buf  = 5 * 4 * 16 * 4    # 1280 bytes
gap_buf    = 16          * 4    # 64   bytes
dense_buf  = 16          * 4    # 64   bytes
output_buf = 1           * 4    # 4    bytes

raw_bytes  = (input_buf + conv1_buf + conv2_buf +
              gap_buf   + dense_buf + output_buf)
overhead   = 1.30
arena_bytes = int(raw_bytes * overhead)
arena_kb    = arena_bytes / 1024.0

print(f"\n    Estimated peak RAM arena : {arena_kb:.2f} KB")
print(f"    Breakdown:")
print(f"      Input  (6x5x1)  : {input_buf}  bytes")
print(f"      Conv1  (6x5x8)  : {conv1_buf}  bytes")
print(f"      Conv2  (5x4x16) : {conv2_buf} bytes")
print(f"      GAP    (16)     : {gap_buf}   bytes")
print(f"      Dense  (16)     : {dense_buf}   bytes")
print(f"      Output (1)      : {output_buf}    bytes")
print(f"      +30%% overhead   : {arena_bytes} bytes total → {arena_kb:.2f} KB")

# ── Hardware Metric 3: Inference Latency (TFLite, PC baseline) ──
# Run TFLite interpreter (not Keras) — this is the exact runtime
# that executes on the ESP32, making it the correct latency proxy.
# PC clock (~3GHz) is ~20-40x faster than ESP32-C3 (~160MHz),
# so we report both the raw PC measurement and an ESP32 estimate.

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale      = input_details[0]["quantization"][0]
input_zero_point = input_details[0]["quantization"][1]

latencies = []

for i in range(len(X_test)):
    # Input shape for 2D-CNN: (1, 6, 5, 1)
    single_sample = X_test[i].astype(np.float32)  # shape: (6, 5, 1)

    # Quantize float32 → int8 using calibrated scale and zero_point.
    # This mirrors exactly what the ESP32 firmware must do before
    # calling interpreter->Invoke().
    if input_scale != 0:
        sample_q = (single_sample / input_scale + input_zero_point)
        sample_q = np.clip(sample_q, -128, 127).astype(np.int8)
    else:
        sample_q = single_sample.astype(np.int8)

    # Add batch dimension: (6,5,1) → (1,6,5,1)
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

# =============================================================
# STEP 6 — SAVE BENCHMARK RESULTS TO JSON
# =============================================================

print("\n[6/6] Saving benchmark results to JSON...")

benchmark_results = {
    "model": "2D-CNN (TinyML — ESP32 Optimised)",
    "dataset": {
        "total_samples"    : len(df),
        "features_used"    : "sample_2 to sample_31 (30 timesteps)",
        "features_dropped" : "sample_0, sample_1 (pre-strike noise)",
        "train_samples"    : len(X_train),
        "val_samples"      : len(X_val),
        "test_samples"     : len(X_test)
    },
    "architecture": {
        "input_shape"      : [6, 5, 1],
        "reshape_logic"    : "30 timesteps → (6 temporal windows) x (5 samples/window) x (1 channel)",
        "reshape_note"     : (
            "Spatial adjacency in the 2D matrix does not equal temporal adjacency "
            "in the original signal. Cross-row temporal continuity is lost. "
            "The 2D-CNN learns spatial texture patterns, not true temporal dynamics."
        ),
        "layers"           : [
            "Conv2D(8 filters, kernel=(3,3), relu, padding=same) → output (6,5,8)",
            "Conv2D(16 filters, kernel=(2,2), relu, padding=valid) → output (5,4,16)",
            "GlobalAveragePooling2D → output (16,)",
            "Dense(16, relu)",
            "Dense(1, sigmoid)"
        ],
        "total_parameters" : int(total_params),
        "optimizer"        : "Adam (lr=0.001)",
        "loss"             : "binary_crossentropy",
        "early_stopping"   : "patience=5, monitor=val_loss"
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
            "saved_as"       : "confusion_matrix_2dcnn.png"
        }
    },
    "hardware_metrics": {
        "6_flash_memory_kb": float(f"{flash_kb:.2f}"),
        "7_peak_ram_kb"    : float(f"{arena_kb:.2f}"),
        "7_ram_breakdown"  : {
            "input_6x5x1_bytes"  : input_buf,
            "conv1_6x5x8_bytes"  : conv1_buf,
            "conv2_5x4x16_bytes" : conv2_buf,
            "gap_16_bytes"       : gap_buf,
            "dense_16_bytes"     : dense_buf,
            "output_1_bytes"     : output_buf,
            "overhead_factor"    : "30%",
            "total_arena_bytes"  : arena_bytes
        },
        "8_inference_latency_ms": {
            "pc_baseline_avg"       : float(f"{avg_latency_ms:.4f}"),
            "pc_baseline_std"       : float(f"{std_latency_ms:.4f}"),
            "esp32_c3_estimated_ms" : f"{avg_latency_ms*25:.1f} – {avg_latency_ms*40:.1f}",
            "note"                  : "PC baseline ~25-40x faster than ESP32-C3 @ 160MHz"
        },
        "9_power_consumption": "N/A - Requires Physical Multimeter"
    },
    "tflite_model": {
        "path"         : TFLITE_PATH,
        "quantization" : "INT8 post-training quantization",
        "input_type"   : "int8",
        "output_type"  : "int8"
    },
    "scaler": {
        "type"          : "MinMaxScaler",
        "fitted_on"     : "training set only (no leakage)",
        "params_saved"  : "scaler_params_2dcnn.json"
    }
}

JSON_PATH = "benchmark_2dcnn_results.json"
with open(JSON_PATH, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print(f"    Benchmark results saved: {JSON_PATH}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — All outputs generated successfully")
print("=" * 60)
print(f"  coconut_2dcnn.tflite          → Flash to ESP32")
print(f"  benchmark_2dcnn_results.json  → Paper metrics")
print(f"  confusion_matrix_2dcnn.png    → Paper figure")
print(f"  scaler_params_2dcnn.json      → Hardcode into firmware")
print("=" * 60)
print(f"\n  QUICK METRICS SUMMARY")
print(f"  Accuracy       : {accuracy:.2f}%")
print(f"  F1-Score       : {f1:.4f}")
print(f"  Flash (KB)     : {flash_kb:.2f}")
print(f"  RAM arena (KB) : {arena_kb:.2f}")
print(f"  Latency PC(ms) : {avg_latency_ms:.4f}")
print("=" * 60)