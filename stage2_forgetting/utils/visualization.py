import matplotlib.pyplot as plt
import os

def plot_accuracy_curve(strengths, cat_acc, dog_acc, bird_acc, save_path='results/accuracy_curve.png'):
    """绘制准确率变化曲线"""
    plt.figure(figsize=(8, 5))
    plt.plot(strengths, cat_acc, label='cat', color='red', marker='o', linewidth=2)
    plt.plot(strengths, dog_acc, label='dog', color='blue', marker='s', linewidth=2)
    plt.plot(strengths, bird_acc, label='bird', color='green', marker='^', linewidth=2)

    plt.xlabel('Surgery Strength')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Surgery Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 准确率曲线已保存至：{save_path}")

def plot_feature_distribution(features, labels, save_path='results/feature_distribution.png'):
    """绘制特征分布（t-SNE 降维可视化）"""
    from sklearn.manifold import TSNE

    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    class_names = ['cat', 'dog', 'bird']

    for idx, name in enumerate(class_names):
        mask = [i for i, l in enumerate(labels) if l == idx]
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=colors[idx], label=name, alpha=0.6, s=50)

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Feature Distribution (t-SNE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 特征分布图已保存至：{save_path}")