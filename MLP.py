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
print("  NDT Coconut Ripeness — DNN (MLP) TinyML Pipeline")
print("=" * 60)

# =============================================================
# STEP 1 — DATA LOADING & PREPROCESSING
# =============================================================

print("\n[1/6] Loading and preprocessing dataset...")

df = pd.read_csv("coconut_training_data.csv")

# Drop pre-strike ambient noise columns (sample_0, sample_1).
# For a DNN this is especially important to acknowledge:
# unlike CNNs which can learn to ignore uninformative regions
# via spatial pooling, a Dense layer assigns a learned weight
# to EVERY input feature. sample_0 and sample_1 (values 10-25,
# random ADC noise) would each consume a full column of the
# 30x16 weight matrix W in Dense layer 1. These weights would
# be trained to approximate zero but still waste capacity,
# add noise to gradients, and slightly increase RAM on the ESP32.
# Dropping them keeps the input vector lean and physically clean.
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
# For a DNN, normalisation is critical because Dense layers use
# dot products: output = W @ x + b. If x contains raw ADC values
# (0-4095), the dot product magnitude is ~4000x larger than with
# normalised inputs [0,1]. This forces the optimizer to find
# very small weight values, slows convergence dramatically, and
# makes the loss landscape poorly conditioned. MinMax scaling
# ensures all 30 input features contribute equally to the dot
# product — none dominates just because of its ADC range.
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)   # shape: (2400, 30)
X_val   = scaler.transform(X_val)         # shape: (300,  30)
X_test  = scaler.transform(X_test)        # shape: (300,  30)

# Save scaler parameters for ESP32 firmware preprocessing.
# The microcontroller must apply: x_norm = (x_raw - data_min) * scale + min
# for each of the 30 features before calling interpreter->Invoke().
scaler_params = {
    "data_min" : scaler.data_min_.tolist(),
    "data_max" : scaler.data_max_.tolist(),
    "scale"    : scaler.scale_.tolist(),
    "min"      : scaler.min_.tolist()
}
with open("scaler_params_dnn.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

# ── FLAT INPUT — NO RESHAPE NEEDED ──
# This is the key architectural difference from the CNNs and LSTM.
# The DNN treats the 30 features as a flat, UNORDERED vector.
# There is no concept of "timestep t comes before timestep t+1"
# in a Dense layer — every input connects to every neuron with
# its own learned weight, regardless of position in the vector.
# X shape remains (batch_size, 30) — no channel dim, no 2D reshape.
# This is both the DNN's strength (simple, fast) and its weakness
# (no inductive bias toward temporal structure, so it must learn
# the decay pattern purely from the amplitude values themselves).

print(f"    Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
print(f"    DNN input: flat 30-element vector (no temporal structure assumed)")
print(f"    Label balance — Train Ripe: {y_train.sum()}  Unripe: {(y_train==0).sum()}")

# =============================================================
# STEP 2 — MODEL ARCHITECTURE
# =============================================================

print("\n[2/6] Building DNN (MLP) architecture...")

# WHY THIS SPECIFIC LAYER CONFIGURATION:
#
# Dense(30 → 16): The first layer compresses the 30 input features
# into 16 learned representations. Each of the 16 neurons computes
# a weighted sum across ALL 30 inputs — it can in principle learn
# "high sample_2 + fast drop by sample_5 = Unripe" as a single
# neuron activation. Parameters: 30*16 + 16 = 496.
#
# Dense(16 → 8): Second layer learns non-linear combinations of
# the 16 first-layer features. This is where the DNN can combine
# "peak amplitude feature" + "mid-decay level feature" into a
# higher-order "decay rate" representation. Parameters: 16*8 + 8 = 136.
#
# Dense(8 → 1): Binary sigmoid output. Parameters: 8*1 + 1 = 9.
#
# Total: 496 + 136 + 9 = 641 parameters.
# This is the SMALLEST parameter count of all 5 models in the
# benchmark — making it the most memory-efficient on the ESP32
# in terms of Flash storage for the model weights.

model = tf.keras.Sequential([

    # Dense layer 1: 16 units, ReLU.
    # input_shape=(30,) — flat vector, no channel/timestep dimension.
    # ReLU: output = max(0, W@x + b). Provides non-linearity while
    # being computationally trivial — one comparison per neuron,
    # no exp() or tanh() calls. This matters on an ESP32 where
    # every floating point operation has a cost.
    tf.keras.layers.Dense(
        16,
        activation="relu",
        input_shape=(30,),
        name="dense_hidden_1"
    ),

    # Dense layer 2: 8 units, ReLU.
    # Further compression from 16 → 8 features. The bottleneck
    # forces the network to distil the most class-discriminating
    # information from the first layer into 8 compact activations.
    # At inference on the ESP32, this layer requires only
    # 16*8 = 128 multiply-accumulate operations — negligible cost.
    tf.keras.layers.Dense(
        8,
        activation="relu",
        name="dense_hidden_2"
    ),

    # Output: single sigmoid neuron.
    # Sigmoid maps the linear combination of 8 hidden features
    # into a probability in (0,1). Threshold at 0.5:
    # > 0.5 → Ripe (1), <= 0.5 → Unripe (0).
    # On the ESP32, the sigmoid can be approximated with a lookup
    # table for faster inference if needed.
    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )
], name="coconut_dnn")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

total_params = model.count_params()
print(f"\n    Total trainable parameters: {total_params:,}")
print(f"    Layer breakdown:")
print(f"      Dense(30→16) : 30*16 + 16 bias = {30*16+16} params")
print(f"      Dense(16→ 8) : 16*8  +  8 bias = {16*8+8}  params")
print(f"      Dense( 8→ 1) :  8*1  +  1 bias = {8*1+1}   params")

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

# Representative dataset for INT8 calibration.
# The quantizer uses these samples to measure the dynamic range
# (min/max) of every activation tensor in the network, then
# computes the INT8 scale and zero_point for each layer.
# For a DNN, this is simpler than LSTM — Dense ops are fully
# supported as TFLITE_BUILTINS_INT8 in all TF versions >= 2.4.
# Input shape: (1, 30) — batch dim + flat feature vector.
def representative_dataset_generator():
    calibration_data = X_train[:200].astype(np.float32)
    for sample in calibration_data:
        # Dense model expects (1, 30) — add batch dimension only.
        # No channel dim, no timestep dim — flat vector as-is.
        yield [np.expand_dims(sample, axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Full INT8 quantization: weights AND activations → int8.
# For Dense layers this is extremely clean:
#   output_int8 = (W_int8 @ input_int8) * scale + zero_point
# No Select TF Ops needed — Dense ops are native TFLite builtins.
# This gives the DNN the best Flash/RAM ratio of all 5 models
# since Dense weight matrices quantize very efficiently to INT8.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

TFLITE_PATH = "coconut_dnn.tflite"
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

flash_kb = os.path.getsize(TFLITE_PATH) / 1024.0
print(f"    TFLite model saved : {TFLITE_PATH}")
print(f"    Flash footprint    : {flash_kb:.2f} KB")

# =============================================================
# STEP 5 — METRIC EVALUATION
# =============================================================

print("\n[5/6] Evaluating all 9 benchmark metrics...")

# ── ML Metrics (Keras float32 model) ──

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
    cmap="Greens",
    xticklabels=["Unripe (0)", "Ripe (1)"],
    yticklabels=["Unripe (0)", "Ripe (1)"],
    ax=ax
)
ax.set_title("Confusion Matrix — DNN/MLP (Test Set)", fontsize=13, pad=12)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label",      fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix_dnn.png", dpi=150)
plt.close()
print("\n    Confusion matrix saved: confusion_matrix_dnn.png")

# ── Hardware Metric 2: Peak RAM Arena Estimate ──
#
# DNN activation buffers are the simplest of all 5 models.
# Unlike CNNs (spatial activation maps) or LSTMs (persistent
# state tensors across timesteps), a DNN only needs RAM for
# the current layer's input and output simultaneously.
#
# TFLite Micro can reuse the same memory buffer for consecutive
# Dense layer outputs because each layer only reads its input
# once before the buffer can be overwritten. The peak memory
# moment is when the largest pair of (input, output) coexist.
#
# Layer-by-layer activation sizes (float32, 4 bytes each):
#   Input        : 30 values  × 4 bytes = 120 bytes
#   Dense-1 out  : 16 values  × 4 bytes =  64 bytes
#   Dense-2 out  :  8 values  × 4 bytes =  32 bytes
#   Output       :  1 value   × 4 bytes =   4 bytes
#
# Peak moment: input(120) + Dense-1 output(64) = 184 bytes
# (both must exist simultaneously while Dense-1 computes)
# We sum ALL buffers conservatively since TFLite Micro arena
# pre-allocates worst-case, then add 30% overhead.

input_buf  = 30 * 4    # 120 bytes — flat input vector
dense1_buf = 16 * 4    # 64  bytes — Dense-1 activations
dense2_buf =  8 * 4    # 32  bytes — Dense-2 activations
output_buf =  1 * 4    # 4   bytes — sigmoid output

raw_bytes   = input_buf + dense1_buf + dense2_buf + output_buf
overhead    = 1.30
arena_bytes = int(raw_bytes * overhead)
arena_kb    = arena_bytes / 1024.0

print(f"\n    DNN Peak RAM estimate:")
print(f"      Input   (30 values) : {input_buf}  bytes")
print(f"      Dense-1 (16 values) : {dense1_buf}   bytes")
print(f"      Dense-2  (8 values) : {dense2_buf}   bytes")
print(f"      Output   (1 value)  : {output_buf}    bytes")
print(f"      Raw total           : {raw_bytes}  bytes")
print(f"      +30%% overhead      : {arena_bytes}  bytes")
print(f"      Estimated arena     : {arena_kb:.2f} KB")
print(f"    NOTE: DNN has lowest RAM of all 5 models.")
print(f"    No spatial maps (CNN) or persistent states (LSTM).")

# ── Hardware Metric 3: Inference Latency (TFLite, PC baseline) ──
# DNN inference is the simplest compute graph of all 5 models:
# three matrix-vector multiplications in sequence.
# Expected to be the fastest model at inference time on the ESP32,
# though the 1D-CNN with GlobalAveragePooling may be competitive
# since its conv ops are also highly optimised in TFLite Micro.

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype      = input_details[0]["dtype"]
input_scale      = input_details[0]["quantization"][0]
input_zero_point = input_details[0]["quantization"][1]

latencies = []

for i in range(len(X_test)):
    # Input shape for DNN: (1, 30) — batch dim + flat vector.
    # No channel dimension, no timestep dimension.
    single_sample = X_test[i].astype(np.float32)   # shape: (30,)

    # Quantize float32 → int8.
    # Formula: q = float_val / scale + zero_point, clamp to [-128,127]
    if input_dtype == np.int8 and input_scale != 0:
        sample_q = (single_sample / input_scale + input_zero_point)
        sample_q = np.clip(sample_q, -128, 127).astype(np.int8)
    elif input_dtype == np.int8:
        sample_q = single_sample.astype(np.int8)
    else:
        sample_q = single_sample   # float32 fallback

    # Add batch dimension: (30,) → (1, 30)
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
print(f"    Expected: fastest or joint-fastest of all 5 models.")

# =============================================================
# STEP 6 — SAVE BENCHMARK RESULTS TO JSON
# =============================================================

print("\n[6/6] Saving benchmark results to JSON...")

benchmark_results = {
    "model": "DNN / MLP (TinyML — ESP32 Optimised)",
    "dataset": {
        "total_samples"    : len(df),
        "features_used"    : "sample_2 to sample_31 (30 features)",
        "features_dropped" : "sample_0, sample_1 (pre-strike noise)",
        "train_samples"    : int(len(X_train)),
        "val_samples"      : int(len(X_val)),
        "test_samples"     : int(len(X_test))
    },
    "architecture": {
        "input_shape"      : [30],
        "input_note"       : (
            "Flat vector — no channel or timestep dimension. "
            "DNN treats all 30 features as an unordered set. "
            "No inductive bias toward temporal structure."
        ),
        "layers"           : [
            "Dense(16, relu, input_shape=(30,)) — 496 params",
            "Dense(8,  relu)                   — 136 params",
            "Dense(1,  sigmoid)                —   9 params"
        ],
        "total_parameters" : int(total_params),
        "param_breakdown"  : {
            "dense_1_30x16_plus_bias" : int(30 * 16 + 16),
            "dense_2_16x8_plus_bias"  : int(16 * 8  + 8),
            "output_8x1_plus_bias"    : int(8  * 1  + 1)
        },
        "optimizer"        : "Adam (lr=0.001)",
        "loss"             : "binary_crossentropy",
        "early_stopping"   : "patience=5, monitor=val_loss",
        "design_note"      : (
            "Intentionally minimal architecture. DNN serves as the "
            "deep learning baseline — demonstrates how much performance "
            "is attributable to temporal structure learning (CNN/LSTM) "
            "vs. simple amplitude feature discrimination (DNN)."
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
            "saved_as"       : "confusion_matrix_dnn.png"
        }
    },
    "hardware_metrics": {
        "6_flash_memory_kb": float(f"{flash_kb:.2f}"),
        "7_peak_ram_kb"    : float(f"{arena_kb:.2f}"),
        "7_ram_breakdown"  : {
            "input_30_bytes"    : input_buf,
            "dense1_16_bytes"   : dense1_buf,
            "dense2_8_bytes"    : dense2_buf,
            "output_1_bytes"    : output_buf,
            "raw_total_bytes"   : raw_bytes,
            "overhead_factor"   : "30%",
            "total_arena_bytes" : arena_bytes,
            "key_note"          : (
                "Lowest RAM of all 5 benchmark models. "
                "No spatial activation maps (CNN) and no "
                "persistent recurrent state tensors (LSTM). "
                "TFLite Micro can reuse activation buffers "
                "between sequential Dense layers."
            )
        },
        "8_inference_latency_ms": {
            "pc_baseline_avg"       : float(f"{avg_latency_ms:.4f}"),
            "pc_baseline_std"       : float(f"{std_latency_ms:.4f}"),
            "esp32_c3_estimated_ms" : f"{avg_latency_ms*25:.1f} – {avg_latency_ms*40:.1f}",
            "note"                  : (
                "Three sequential matrix-vector multiplications. "
                "Simplest compute graph of all 5 models. "
                "Expected lowest or joint-lowest latency on ESP32."
            )
        },
        "9_power_consumption": "N/A - Requires Physical Multimeter"
    },
    "tflite_model": {
        "path"         : TFLITE_PATH,
        "quantization" : "INT8 post-training quantization (standard builtins)",
        "input_type"   : "int8",
        "output_type"  : "int8",
        "esp32_note"   : (
            "Dense ops are fully supported as TFLITE_BUILTINS_INT8. "
            "No SELECT_TF_OPS required. Compatible with "
            "MicroMutableOpResolver with only AddFullyConnected()."
        )
    },
    "scaler": {
        "type"         : "MinMaxScaler",
        "fitted_on"    : "training set only (no leakage)",
        "params_saved" : "scaler_params_dnn.json"
    }
}

JSON_PATH = "benchmark_dnn_results.json"
with open(JSON_PATH, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print(f"    Benchmark results saved: {JSON_PATH}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE — All outputs generated successfully")
print("=" * 60)
print(f"  coconut_dnn.tflite            → Flash to ESP32")
print(f"  benchmark_dnn_results.json    → Paper metrics")
print(f"  confusion_matrix_dnn.png      → Paper figure")
print(f"  scaler_params_dnn.json        → Hardcode into firmware")
print("=" * 60)
print(f"\n  QUICK METRICS SUMMARY")
print(f"  Accuracy         : {accuracy:.2f}%")
print(f"  F1-Score         : {f1:.4f}")
print(f"  Flash (KB)       : {flash_kb:.2f}")
print(f"  RAM arena (KB)   : {arena_kb:.2f}")
print(f"  Latency PC (ms)  : {avg_latency_ms:.4f}")
print(f"  Parameters       : {total_params} (fewest of all 5 models)")
print("=" * 60)