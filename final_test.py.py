# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

file_path = r"C:\Users\ADMIN\Documents\adhdata.csv"
data = pd.read_csv(file_path)
X = data.iloc[:, :19].values  
y = data['Class'].values  

y = np.where(y == 'ADHD', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std


X_train = X_train.reshape(-1, 19, 1)
X_test = X_test.reshape(-1, 19, 1)

model = keras.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(19, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    
    layers.Conv1D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)


history = model.fit(
    X_train, y_train,
    validation_split=0.2,  
    epochs=5,
    batch_size=13200,
    verbose=1
)

model.save('newad.h5')

from tensorflow.keras.models import load_model
model=load_model('newad.h5')


## 
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, y_pred)

cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

labels = np.empty_like(cm, dtype=object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = cm_percentage[i, j]
        labels[i, j] = f'{count}\n({percentage:.1f}%)'


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
            xticklabels=['Healthy', 'ADHD'], 
            yticklabels=['Healthy', 'ADHD'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 12, 'weight': 'bold', 'va': 'center', 'ha': 'center'})

plt.title('Confusion Matrix - ADHD Detection\n(Count with Percentage)', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Healthy', 'ADHD'], 
            yticklabels=['Healthy', 'ADHD'])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Reds', ax=axes[1],
            xticklabels=['Healthy', 'ADHD'], 
            yticklabels=['Healthy', 'ADHD'])
axes[1].set_title('Confusion Matrix (Percentage %)', fontsize=14)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.suptitle('ADHD Detection Performance', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

TN, FP, FN, TP = cm.ravel()

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))

plt.hist(y_pred_prob[y_test == 0], bins=30, alpha=0.5, label='Healthy (Actual)', color='blue')
plt.hist(y_pred_prob[y_test == 1], bins=30, alpha=0.5, label='ADHD (Actual)', color='red')

plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

