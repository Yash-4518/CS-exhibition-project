import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
WEBCAM_CSV  = "/Users/yashsharma/Desktop/Drowsiness/webcam_data.csv"
MODEL_OUT   = "/Users/yashsharma/Desktop/Drowsiness/drowsiness_model.h5"
SCALER_OUT  = "/Users/yashsharma/Desktop/Drowsiness/scaler.pkl"

# ─────────────────────────────────────────────
#  Load webcam data
# ─────────────────────────────────────────────
df = pd.read_csv(WEBCAM_CSV)
print(f"Webcam data loaded: {len(df)} samples")
print(f"  Alert  (0): {(df['label']==0).sum()}")
print(f"  Drowsy (1): {(df['label']==1).sum()}")

X_webcam = df.drop('label', axis=1).values.astype(np.float32)
y_webcam = df['label'].values.astype(np.float32)

# ─────────────────────────────────────────────
#  Scale using existing scaler (keeps consistency
#  with original dataset distribution)
# ─────────────────────────────────────────────
scaler = joblib.load(SCALER_OUT)
X_scaled = scaler.transform(X_webcam)

# ─────────────────────────────────────────────
#  Split
# ─────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_webcam, test_size=0.2, random_state=42, stratify=y_webcam
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

# ─────────────────────────────────────────────
#  Load existing model and fine-tune on webcam data
# ─────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_OUT)

# lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ─────────────────────────────────────────────
#  Save fine-tuned model
# ─────────────────────────────────────────────
model.save(MODEL_OUT)
print(f"\nFine-tuned model saved → {MODEL_OUT}")

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy : {val_acc*100:.2f}%")

plt.plot(history.history['accuracy'],     label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Fine-tuning accuracy'); plt.legend()
plt.savefig("/Users/yashsharma/Desktop/Drowsiness/finetune_curves.png")
plt.show()
