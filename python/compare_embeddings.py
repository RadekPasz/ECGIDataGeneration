import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DISCRIM_PATH  = 'python/discriminator_advanced.pth'
REAL_DIR      = 'gan_segments'
SYNTH_DIR     = 'synth_samples'
SEL_BSP_CH    = 0    
SEL_HSP_ND    = 0
N_SAMPLES     = 500  
TEST_SIZE     = 0.2  
K_NEIGHBORS   = 3    

#Load discriminator and extract feature layers
from train_cgan import Discriminator
D = Discriminator().to(DEVICE)
D.load_state_dict(torch.load(DISCRIM_PATH, map_location=DEVICE))
D.eval()
feature_extractor = torch.nn.Sequential(D.layer1, D.layer2, D.layer3).to(DEVICE)

#Embedding function
def get_embedding(npz_path):
    data = np.load(npz_path)
    bsp_arr = data['bsp']
    if bsp_arr.ndim > 1:
        lead = bsp_arr[:, SEL_BSP_CH]
    else:
        lead = bsp_arr
    bsp = torch.from_numpy(lead).float().unsqueeze(0)
    zeros = torch.zeros_like(bsp)
    x = torch.stack([bsp, zeros], dim=1).to(DEVICE)
    with torch.no_grad():
        feats = feature_extractor(x)        
        vec   = feats.mean(dim=2).squeeze(0) 
    return vec.cpu().numpy()

#Gather embeddings and labels
real_paths  = sorted(Path(REAL_DIR).rglob('window*.npz'))[:N_SAMPLES]
synth_paths = sorted(Path(SYNTH_DIR).rglob('synth_*.npz'))[:N_SAMPLES]
embs, labels = [], []
for p in real_paths:
    embs.append(get_embedding(str(p)))
    labels.append(0)
for p in synth_paths:
    embs.append(get_embedding(str(p)))
    labels.append(1)
embs = np.stack(embs)
labels = np.array(labels)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
proj = tsne.fit_transform(embs)
plt.figure(figsize=(6,6))
plt.scatter(proj[labels==0,0], proj[labels==0,1], c='C0', alpha=0.7, label='Real')
plt.scatter(proj[labels==1,0], proj[labels==1,1], c='C1', alpha=0.7, label='Synth')
plt.legend(); plt.title('t-SNE: Real vs Synthetic Embeddings')
plt.xlabel('Dim 1'); plt.ylabel('Dim 2'); plt.tight_layout(); plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    embs, labels, test_size=TEST_SIZE, random_state=42, stratify=labels)
knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"KNN classification accuracy: {acc*100:.2f}%")
print("Classification report:\n", classification_report(y_test, y_pred, target_names=['Real','Synth']))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
labels = ['Real', 'Synth']

fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
fig.colorbar(im, ax=ax)

#Tick labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

#Annotate cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="black" if cm[i, j] > thresh else "black")

ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
ax.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()
