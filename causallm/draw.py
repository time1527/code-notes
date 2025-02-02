import pandas as pd
import matplotlib.pyplot as plt
from config import PretrainConfig, SFTConfig


def plot_loss_from_csv(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 假设每一行数据代表 10 次迭代
    loss = df["0"]  # 使用列名 '0' 来获取损失数据
    iterations = [i * 10 for i in range(len(loss))]  # 生成迭代次数

    # 获取文件名并去掉扩展名
    plot_filename = csv_path.split("/")[-1].split(".")[0]
    if "pretrain" in plot_filename.lower():
        li = PretrainConfig.log_interval
    elif "sft" in plot_filename.lower():
        li = SFTConfig.log_interval

    # 绘制折线图
    plt.plot(iterations, loss, label="Loss", color="blue")
    plt.xlabel(f"#iterations:log-intervals = {li}")  # 横轴标签
    plt.ylabel("Loss")  # 纵轴标签
    plt.title(f"{plot_filename}")  # 图表标题
    plt.grid(True)  # 显示网格
    plt.legend()

    plt.savefig(f".assets/{plot_filename}.png")  # 保存图像为文件
    plt.show()  # 显示图表


if __name__ == "__main__":
    plot_loss_from_csv("saves/pretrain-loss.csv")
    plot_loss_from_csv("saves/sft-loss.csv")
