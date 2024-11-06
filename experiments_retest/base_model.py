import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


from liquid_dataset.gesture import GestureData
from liquid_dataset.ozone_dataloader import OzoneDataLoader
from experiments_retest.session import Session
from liquidnet.dynamic_rnn import DynamicRNN
import wandb
from liquidnet.liquid_cell import LiquidCell

# 初始化 wandb 项目
wandb.login(key='da3d4988444edcd2ed47373e2c19c6b7fdccd4a7')
wandb.init(project="ltc", name="pytorch-ltc-fine-tune3")


class BaseModel():
    def __init__(self, input_size, model_type, model_size, output_size,seq_len,lr,wand,device):
        super(BaseModel, self).__init__()
        self.model_type = model_type
        self.model_size = model_size
        self.input_size = input_size
        self.device = device
        self.lr = lr
        self.seq_len = seq_len
        self.wandb = wand

        # 定义输出层
        self.constrain_op = None
        self.rnn = self.build_model(model_type=model_type,input_size=input_size,model_size=model_size,output_size=output_size,device=device)
        if model_type == "ltc":
            self.constrain_op = self.rnn.cell.get_param_constrain_op
        else:
            self.constrain_op = None
        # 初始化学习率调度器
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=lr,betas=(0.9, 0.999))
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
        #                                    verbose=True)
        self.criterion = nn.CrossEntropyLoss()
        self.session = Session(model=self.rnn, optimizer=self.optimizer, criterion=self.criterion, constrain_op = self.constrain_op,device=device)
        # 创建保存目录
        result_file = os.path.join("results", f"{self.model_type}_{self.model_size}.csv")
        os.makedirs("results", exist_ok=True)
        self.result_file = result_file
        # 使用 wandb 自动跟踪模型梯度和权重
        self.wandb.watch(self.rnn, criterion=self.criterion, log="all", log_freq=10)

        # 保存 CSV 文件头
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write("epoch,train_loss,train_acc,valid_loss,valid_acc\n")

    def build_model(self,model_type,input_size,model_size,output_size,device):
        # 定义 RNN 层：支持 LSTM 或 GRU（模拟 LTC）
        if model_type == "lstm":
            rnncell = nn.LSTMCell(input_size=input_size, hidden_size=model_size,device=device)
            rnn = DynamicRNN(cell=rnncell,model_type='lstm',output_size=output_size)

        elif model_type == "ltc":
            rnncell = LiquidCell(input_size = input_size,num_units=model_size,device=device)

            rnn = DynamicRNN(cell=rnncell,model_type='ltc',output_size=output_size)
            self.constrain_op = rnncell.get_param_constrain_op
        elif model_type == "rnn":
            rnn = nn.RNN(input_size=input_size, hidden_size=model_size,device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return rnn

    def build_state(self, batch_size):
        if self.model_type == "lstm":
            zeros = torch.zeros(1, batch_size, self.model_size, device=self.device).detach()
            state = (zeros, zeros)
        else:
            state = torch.zeros(batch_size, self.model_size, device=self.device).detach()
        return state
    # 训练函数：增加早停、日志保存和参数约束
    def fit(self, data_loader, epochs, log_period=10,batch_size=16):

        best_accuracy = 0
        best_valid_stats = (0, 0, 0, 0, 0)  # 不再记录test_loss和test_acc

        # 初始化状态
        # state = self.build_state(data_loader.batch_size)

        # 开始训练
        for epoch in range(epochs):
            self.rnn.train()
            losses, accs = [], []

            for batch_x, batch_y in data_loader.iterate_train(batch_size):
                loss, acc = self.session.train_batch(batch_x, batch_y)
                losses.append(loss)
                accs.append(acc)

            avg_train_loss = np.mean(losses)
            avg_train_acc = np.mean(accs)

            if epoch % log_period == 0:
                valid_acc, valid_loss = self.session.evaluate(data_loader=data_loader, split='valid')
                # if self.scheduler:
                #     self.scheduler.step(valid_loss)  # 更新学习率基于验证损失
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch:03d}, Train Loss: {avg_train_loss:.2f}, "
                      f"Train Acc: {avg_train_acc:.2f}%, Valid Loss: {valid_loss:.2f}, "
                      f"Valid Acc: {valid_acc :.2f}%")

                # 将指标记录到 wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "train_accuracy": avg_train_acc,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_acc,
                    "learning_rate": current_lr
                })

                # 如果验证集准确率提升，则保存模型
                if valid_acc > best_accuracy:
                    best_accuracy = valid_acc
                    best_valid_stats = (epoch, avg_train_loss, avg_train_acc, valid_loss, valid_acc)
                    self.session.save_model()

                # 记录结果到 CSV 文件
                with open(self.result_file, "a") as f:
                    f.write(f"{epoch},{avg_train_loss:.2f},{avg_train_acc:.2f},"
                            f"{valid_loss:.2f},{valid_acc:.2f}\n")
            # 终止训练条件
            # print('avg_train_loss:',avg_train_loss)
            # print('np.isfinite(avg_train_loss):',np.isfinite(avg_train_loss))
            if epoch > 0 and not np.isfinite(avg_train_loss):
                print("Training stopped due to non-finite loss.")
                break
            # # 在每个 epoch 结束时记录当前学习率
            # for param_group in optimizer.param_groups:
            #     print(f"Epoch {epoch}: Learning rate = {param_group['lr']}")
        # 打印最佳模型信息
        best_epoch, train_loss, train_acc, valid_loss, valid_acc = best_valid_stats
        print(f"Training completed. Best epoch {best_epoch:03d}, Train Loss: {train_loss:.2f}, "
              f"Train Acc: {train_acc:.2f}%, Valid Loss: {valid_loss:.2f}, "
              f"Valid Acc: {valid_acc:.2f}%")

        self.session.load_model()  # 加载最佳模型
        test_acc, test_loss = self.session.evaluate(data_loader=data_loader, split='test')
        print(f"test Loss: {test_loss:.2f}, test Acc: {test_acc :.2f}%")
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})



def get_data_loader(data_type, seq_len,batch_size,data_path, device):
    if data_type == "ozone":
        return OzoneDataLoader(seq_len=seq_len,path=data_path, device=device,batch_size=batch_size)
    elif data_type == "gesture":
        ges_dataloader = GestureData(seq_len=seq_len, path=data_path,batch_size=batch_size,device=device)
        return ges_dataloader
        # return GestureDataLoader(seq_len=seq_len,path=data_path, device=device,batch_size=batch_size)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def summarize_gesture_dataloader(dataloader):
    def summary_stats(tensor):
        return {
            "min": float(torch.min(tensor)),
            "max": float(torch.max(tensor)),
            "mean": float(torch.mean(tensor))
        }

    train_mean_stats = summary_stats(torch.mean(dataloader.train_x, dim=(0, 1)))
    train_std_stats = summary_stats(torch.std(dataloader.train_x, dim=(0, 1)))
    valid_mean_stats = summary_stats(torch.mean(dataloader.valid_x, dim=(0, 1)))
    valid_std_stats = summary_stats(torch.std(dataloader.valid_x, dim=(0, 1)))
    test_mean_stats = summary_stats(torch.mean(dataloader.test_x, dim=(0, 1)))
    test_std_stats = summary_stats(torch.std(dataloader.test_x, dim=(0, 1)))

    summary = {
        "total_sequences": {
            "train": dataloader.train_x.size(1),
            "valid": dataloader.valid_x.size(1),
            "test": dataloader.test_x.size(1),
        },
        "feature_statistics": {
            "train": {"mean_stats": train_mean_stats, "std_stats": train_std_stats},
            "valid": {"mean_stats": valid_mean_stats, "std_stats": valid_std_stats},
            "test": {"mean_stats": test_mean_stats, "std_stats": test_std_stats},
        },
        "label_distribution": {
            "train": torch.bincount(dataloader.train_y.flatten()).tolist(),
            "valid": torch.bincount(dataloader.valid_y.flatten()).tolist(),
            "test": torch.bincount(dataloader.test_y.flatten()).tolist(),
        },
    }
    return summary
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['ozone', 'gesture'], required=True, help='选择数据类型',default='ozone')
    parser.add_argument('--data_path', required=True, help='数据路径',default='/mnt/f/dataset/liquid_data/ozone/eighthr.data')
    parser.add_argument('--model_type', choices=['lstm', 'ltc','rnn'], default='lstm', help='模型类型')
    parser.add_argument('--input_size', type=int, default=32, help='输入特征维度')
    parser.add_argument('--model_size', type=int, default=64, help='RNN 隐藏层大小')
    parser.add_argument('--output_size', type=int, default=2, help='输出类别数量')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--seq_len', type=int, default=32, help='序列长度')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    # 在脚本开头设置随机数种子
    set_seed(seed=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 记录超参数
    wandb.config.update({
        "model_type": args.model_type,
        "input_size": args.input_size,
        "model_size": args.model_size,
        "output_size": args.output_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.lr
    })

    # 获取数据加载器
    data_loader = get_data_loader(data_type=args.data_type,seq_len=args.seq_len,
                                  data_path= args.data_path, device=device,batch_size=args.batch_size)

    # aa = summarize_gesture_dataloader(data_loader)
    # print(aa)
    # 初始化模型
    model = BaseModel(input_size=args.input_size, model_type=args.model_type,
                      model_size=args.model_size, output_size=args.output_size,
                      device=device,seq_len=args.seq_len,lr=args.lr,wand=wandb)

    # 训练模型
    state = model.fit(data_loader=data_loader, epochs=args.epochs,log_period=10)


    path = '/mnt/data/gesture'
