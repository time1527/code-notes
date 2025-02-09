import json
import matplotlib.pyplot as plt


data = json.load(open("./saves/results.json"))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(data["train_losses"], label="Train Loss")
plt.plot(data["test_losses"], label="Test Loss")
plt.title("Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 绘制训练和测试准确率
plt.subplot(1, 2, 2)
plt.plot(data["train_accs"], label="Train Accuracy")
plt.plot(data["test_accs"], label="Test Accuracy")
plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# 找到测试准确率的最高值及其索引
max_test_acc = max(data["test_accs"])
max_test_acc_index = data["test_accs"].index(max_test_acc)

# 在图中添加横线和最高值的标注
plt.axhline(
    y=max_test_acc,
    color="r",
    linestyle="--",
    label=f"Max Test Accuracy: {max_test_acc:.2f}",
)
plt.text(
    max_test_acc_index, max_test_acc, f"{max_test_acc:.2f}", ha="center", va="bottom"
)

plt.legend()
plt.savefig("./saves/results.png")
plt.tight_layout()
plt.show()
