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
print("  NDT Coconut Ripeness — 1D-CNN TinyML Pipeline")
print("=" * 60)

# =============================================================
# STEP 1 — DATA LOADING & PREPROCESSING
# =============================================================

print("\n[1/6] Loading and preprocessing dataset...")

df = pd.read_csv("coconut_training_data.csv")

# Drop pre-strike ambient noise columns (sample_0, sample_1).
# These carry no class-discriminating information — only
# random ADC noise between 10-25. Keeping them adds spurious
# signal and wastes one conv filter pass on garbage data.
df = df.drop(columns=["sample_0", "sample_1"])

# Encode labels: Unripe=0, Ripe=1.
# Binary integer encoding is required for sigmoid output +
# binary_crossentropy loss function.
df["label"] = df["label"].map({"Unripe": 0, "Ripe": 1})

X = df.drop(columns=["label"]).values   # shape: (3000, 30)
y = df["label"].values                  # shape: (3000,)

# Split: 80% train, 10% validation, 10% test.
# Stratify ensures class balance is preserved in every split.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# MinMaxScaler: fit ONLY on training data to prevent data leakage.
# If we fit on the full dataset, the model indirectly sees test
# set statistics during training — a critical research error.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Save scaler parameters for hardcoding into ESP32 firmware.
# The ESP32 must apply identical normalisation before inference.
scaler_params = {
    "data_min":  scaler.data_min_.tolist(),
    "data_max":  scaler.data_max_.tolist(),
    "scale":     scaler.scale_.tolist(),
    "min":       scaler.min_.tolist()
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

# Reshape for 1D-CNN: (batch, timesteps, channels).
# 30 timesteps = sample_2 through sample_31.
# 1 channel = single piezo sensor (univariate time-series).
X_train = X_train.reshape(-1, 30, 1)
X_val   = X_val.reshape(-1, 30, 1)
X_test  = X_test.reshape(-1, 30, 1)

print(f"    Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"    Label distribution — Train Ripe: {y_train.sum()}  Unripe: {(y_train==0).sum()}")

# =============================================================
# STEP 2 — MODEL ARCHITECTURE
# =============================================================

print("\n[2/6] Building 1D-CNN architecture...")

model = tf.keras.Sequential([

    # Conv layer 1: 16 filters, kernel=5.
    # A kernel of 5 spans 5 consecutive ADC samples — wide
    # enough to capture the peak at sample_2 AND the first
    # steep drop of the Unripe fast-decay simultaneously.
    tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        activation="relu",
        padding="same",
        input_shape=(30, 1),
        name="conv1d_peak_detector"
    ),

    # Conv layer 2: 32 filters, kernel=3.
    # Narrower kernel detects fine-grained slope changes in
    # the mid-decay region — the slow Ripe ring vs fast drop.
    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="conv1d_decay_detector"
    ),

    # GlobalAveragePooling instead of Flatten.
    # Flatten(30 * 32) = 960 values fed into Dense.
    # GAP averages each of the 32 feature maps into 1 value = 32 values.
    # This reduces Dense layer parameters by 30x, saving critical
    # ESP32 RAM without sacrificing accuracy on this signal type.
    tf.keras.layers.GlobalAveragePooling1D(
        name="global_avg_pool"
    ),

    # Dense 16: learns non-linear combinations of the pooled
    # conv features — amplitude class + decay rate class.
    tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_classifier"
    ),

    # Output: single sigmoid unit for binary classification.
    # Output > 0.5 → Ripe (1), Output <= 0.5 → Unripe (0).
    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )
], name="coconut_1dcnn")

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
    restore_best_weights=True,  # Roll back to best val_loss epoch
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

print(f"\n    Training stopped at epoch: {len(history.history['loss'])}")
print(f"    Best val_loss            : {min(history.history['val_loss']):.4f}")
print(f"    Best val_accuracy        : {max(history.history['val_accuracy']):.4f}")

# =============================================================
# STEP 4 — TFLITE CONVERSION & QUANTIZATION
# =============================================================

print("\n[4/6] Converting to TFLite with post-training quantization...")

# Representative dataset generator.
# The TFLite quantizer uses this to calibrate INT8 scale factors
# for every tensor in the model. Must use TRAINING data only —
# never test data. 100-200 samples is sufficient for calibration.
def representative_dataset_generator():
    calibration_data = X_train[:200].astype(np.float32)
    for sample in calibration_data:
        # TFLite expects shape (1, 30, 1) — add batch dimension
        yield [np.expand_dims(sample, axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Full INT8 quantization: weights AND activations converted.
# This is required for maximum speed on ESP32 which has no
# hardware floating-point unit — INT8 ops run ~3-5x faster.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

TFLITE_PATH = "coconut_1dcnn.tflite"
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

flash_kb = os.path.getsize(TFLITE_PATH) / 1024.0
print(f"    TFLite model saved : {TFLITE_PATH}")
print(f"    Flash footprint    : {flash_kb:.2f} KB")

# =============================================================
# STEP 5 — METRIC EVALUATION
# =============================================================

print("\n[5/6] Evaluating all 9 benchmark metrics...")

# ── ML Metrics (run on float32 Keras model for clean probabilities) ──

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
    cmap="Blues",
    xticklabels=["Unripe (0)", "Ripe (1)"],
    yticklabels=["Unripe (0)", "Ripe (1)"],
    ax=ax
)
ax.set_title("Confusion Matrix — 1D-CNN (Test Set)", fontsize=13, pad=12)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label",      fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix_1dcnn.png", dpi=150)
plt.close()
print("\n    Confusion matrix saved: confusion_matrix_1dcnn.png")

# ── Hardware Metric 1: Flash Memory ──
# Already calculated above from actual .tflite file size.

# ── Hardware Metric 2: Peak RAM (Arena Size Estimate) ──
# The TFLite Micro tensor arena must hold:
#   (a) All model weights (already in Flash, NOT RAM on ESP32)
#   (b) Input tensor buffer
#   (c) All intermediate activation tensors simultaneously
#   (d) Output tensor buffer
# We estimate by summing the largest simultaneous activation buffers.
# Formula: input(30*1*4) + conv1_out(30*16*4) + conv2_out(30*32*4)
#          + gap_out(32*4) + dense_out(16*4) + output(1*4) bytes
# Multiply by 4 for float32 (even after INT8 quant, arena is
# allocated in float32 units in many TFLite Micro implementations).
# Add 30% overhead for TFLite Micro internal bookkeeping.

input_buf    = 30 * 1  * 4
conv1_buf    = 30 * 16 * 4
conv2_buf    = 30 * 32 * 4
gap_buf      = 32      * 4
dense_buf    = 16      * 4
output_buf   = 1       * 4
raw_bytes    = input_buf + conv1_buf + conv2_buf + gap_buf + dense_buf + output_buf
overhead     = 1.30
arena_bytes  = int(raw_bytes * overhead)
arena_kb     = arena_bytes / 1024.0

print(f"\n    Estimated peak RAM arena : {arena_kb:.2f} KB")

# ── Hardware Metric 3: Inference Latency (PC baseline) ──
# We run the TFLite interpreter (not Keras) for accurate latency.
# This simulates exactly what the ESP32 runtime will execute,
# minus the difference in clock speed (PC ~3GHz vs ESP32 ~160MHz).
# The PC baseline is ~20-50x faster than real ESP32 latency.

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = (
    input_details[0]["quantization"][0],
    input_details[0]["quantization"][1]
)

latencies = []

for i in range(len(X_test)):
    single_sample = X_test[i].astype(np.float32)

    # Quantize float32 input → INT8 to match converter settings.
    # Formula: q = float_value / scale + zero_point
    if input_scale != 0:
        single_sample_q = (single_sample / input_scale + input_zero_point)
        single_sample_q = np.clip(single_sample_q, -128, 127).astype(np.int8)
    else:
        single_sample_q = single_sample.astype(np.int8)

    input_data = np.expand_dims(single_sample_q, axis=0)  # (1, 30, 1)

    interpreter.set_tensor(input_details[0]["index"], input_data)

    t_start = time.perf_counter()
    interpreter.invoke()
    t_end   = time.perf_counter()

    latencies.append((t_end - t_start) * 1000.0)  # convert to ms

avg_latency_ms = np.mean(latencies)
std_latency_ms = np.std(latencies)
print(f"\n    Avg inference latency (PC baseline) : {avg_latency_ms:.4f} ms")
print(f"    Std deviation                        : {std_latency_ms:.4f} ms")
print(f"    Note: ESP32-C3 @ 160MHz estimated    : {avg_latency_ms * 25:.1f} – {avg_latency_ms * 40:.1f} ms")

# =============================================================
# STEP 6 — SAVE BENCHMARK RESULTS TO JSON
# =============================================================

print("\n[6/6] Saving benchmark results to JSON...")

benchmark_results = {
    "model": "1D-CNN (TinyML — ESP32 Optimised)",
    "dataset": {
        "total_samples"    : len(df),
        "features_used"    : "sample_2 to sample_31 (30 timesteps)",
        "features_dropped" : "sample_0, sample_1 (pre-strike noise)",
        "train_samples"    : len(X_train),
        "val_samples"      : len(X_val),
        "test_samples"     : len(X_test)
    },
    "architecture": {
        "input_shape"       : [30, 1],
        "layers"            : [
            "Conv1D(16 filters, kernel=5, relu, padding=same)",
            "Conv1D(32 filters, kernel=3, relu, padding=same)",
            "GlobalAveragePooling1D",
            "Dense(16, relu)",
            "Dense(1, sigmoid)"
        ],
        "total_parameters"  : int(total_params),
        "optimizer"         : "Adam (lr=0.001)",
        "loss"              : "binary_crossentropy",
        "early_stopping"    : "patience=5, monitor=val_loss"
    },
    "training": {
        "epochs_trained"    : len(history.history["loss"]),
        "best_val_loss"     : float(f"{min(history.history['val_loss']):.4f}"),
        "best_val_accuracy" : float(f"{max(history.history['val_accuracy']):.4f}")
    },
    "ml_metrics": {
        "1_accuracy_pct"    : float(f"{accuracy:.2f}"),
        "2_precision"       : float(f"{precision:.4f}"),
        "3_recall"          : float(f"{recall:.4f}"),
        "4_f1_score"        : float(f"{f1:.4f}"),
        "5_confusion_matrix": {
            "true_negative"  : int(cm[0][0]),
            "false_positive" : int(cm[0][1]),
            "false_negative" : int(cm[1][0]),
            "true_positive"  : int(cm[1][1]),
            "saved_as"       : "confusion_matrix_1dcnn.png"
        }
    },
    "hardware_metrics": {
        "6_flash_memory_kb" : float(f"{flash_kb:.2f}"),
        "7_peak_ram_kb"     : float(f"{arena_kb:.2f}"),
        "8_inference_latency_ms": {
            "pc_baseline_avg"        : float(f"{avg_latency_ms:.4f}"),
            "pc_baseline_std"        : float(f"{std_latency_ms:.4f}"),
            "esp32_c3_estimated_ms"  : f"{avg_latency_ms*25:.1f} – {avg_latency_ms*40:.1f}",
            "note"                   : "PC baseline ~25-40x faster than ESP32-C3 @ 160MHz"
        },
        "9_power_consumption": "N/A - Requires Physical Multimeter"
    },
    "tflite_model": {
        "path"              : TFLITE_PATH,
        "quantization"      : "INT8 post-training quantization",
        "input_type"        : "int8",
        "output_type"       : "int8"
    },
    "scaler": {
        "type"              : "MinMaxScaler",
        "fitted_on"         : "training set only (no leakage)",
        "params_saved_to"   : "scaler_params.json"
    }
}

JSON_PATH = "benchmark_1dcnn_results.json"
with open(JSON_PATH, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print(f"    Benchmark results saved: {JSON_PATH}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — All outputs generated successfully")
print("=" * 60)
print(f"  coconut_1dcnn.tflite          → Flash to ESP32")
print(f"  benchmark_1dcnn_results.json  → Paper metrics")
print(f"  confusion_matrix_1dcnn.png    → Paper figure")
print(f"  scaler_params.json            → Hardcode into firmware")
print("=" * 60)
print(f"\n  QUICK METRICS SUMMARY")
print(f"  Accuracy       : {accuracy:.2f}%")
print(f"  F1-Score       : {f1:.4f}")
print(f"  Flash (KB)     : {flash_kb:.2f}")
print(f"  RAM arena (KB) : {arena_kb:.2f}")
print(f"  Latency PC(ms) : {avg_latency_ms:.4f}")
print("=" * 60)