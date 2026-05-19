import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from datasets import generate_datasets
from main import define_models

BASE_DIR = "animations"
os.makedirs(BASE_DIR, exist_ok=True)
cm = LinearSegmentedColormap.from_list("BlueOrange", ["#4B8BBE", "#FCEFDD", "#F29D4B"])


def prob_from_logits(logits):
    if logits.ndim == 2 and logits.shape[1] == 2:
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e[:, 1] / np.sum(e, axis=1)
    return 1 / (1 + np.exp(-logits.flatten()))


models = define_models()
data_sets = generate_datasets()
EPOCHS = 1000
fps= 20
for model in models:
    for X_train, y_train, X_test, y_test, ds_name in data_sets:
        model.clear()
        h = 0.05
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]

        fig, ax = plt.subplots(figsize=(6, 6))

        def update(epoch):
            model.train(X_train, y_train, X_test, y_test, epochs=1, batch_size=32)
            probs = prob_from_logits(model.forward(grid)).reshape(xx.shape)
            ax.clear()
            ax.contourf(xx, yy, probs, levels=50, cmap=cm, vmin=0, vmax=1)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='white', linewidth=0.5, s=20)
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, marker='o', edgecolors='black', linewidth=1, s=30)
            ax.set_title(f"{model.name} - {ds_name} - epoch {epoch+1}")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

        SAVE_DIR = os.path.join(BASE_DIR, ds_name)
        os.makedirs(SAVE_DIR, exist_ok=True)
        ani = FuncAnimation(fig, update, frames=EPOCHS, interval=50, blit=False)
        out = os.path.join(SAVE_DIR, f"{ds_name}_{model.name}.mp4")
        try:
            ani.save(out, writer=FFMpegWriter(fps=fps))
            print("Saved:", out)
        except Exception as e:
            print("Could not save video:", e)
            plt.show()
        plt.close(fig)
