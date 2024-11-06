import torch
from sklearn.metrics import classification_report
from torch import nn


class Session:
    def __init__(self, model, optimizer, criterion, model_path="best_model.pth",constrain_op=None,device="cpu"):
        """
        初始化 PyTorchSession。
        参数：
            model: 要训练的 PyTorch 模型。
            optimizer: 优化器实例。
            criterion: 损失函数实例。
            device: 使用的设备（CPU 或 GPU）。
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_path = model_path  # 模型保存路径
        self.constrain_op = constrain_op

    def compute_loss_and_accuracy(self, outputs, targets):
        """
        计算损失和准确率。
        参数：
            outputs: 模型输出，形状为 [batch_size * seq_len, num_classes]
            targets: 真实标签，形状为 [batch_size * seq_len]
        返回：
            loss: 计算的损失
            accuracy: 计算的准确率
        """
        loss = self.criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)
        correct_predictions = (predictions == targets).float().sum().item()
        accuracy = correct_predictions / targets.size(0) * 100  # 使用正确率计算而非比率
        return loss, accuracy

    def flatten_outputs_and_labels(self, outputs, labels):
        """
        展平模型输出和标签。
        参数：
            outputs: 模型输出，形状为 [batch_size, seq_len, num_classes]
            labels: 真实标签，形状为 [batch_size, seq_len]
        返回：
            展平后的 outputs 和 labels
        """
        outputs = outputs.view(-1, outputs.size(-1))  # [total_samples, num_classes]
        labels = labels.view(-1)  # [total_samples]
        return outputs, labels

    def train_batch(self, batch_x, batch_y, state=None):
        """
        训练单个 batch，计算 loss 和 accuracy。
        """
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        # 前向传播
        outputs,_ = self.model(batch_x, state)  # LSTM 输出

        # 展平输出和标签
        outputs, batch_y = self.flatten_outputs_and_labels(outputs, batch_y)

        # 计算 loss 和 accuracy
        loss, accuracy = self.compute_loss_and_accuracy(outputs, batch_y)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # torch.nn.utils.clip_grad_norm_(self.model.cell.parameters(), max_norm=1.0)
        if self.constrain_op is not None:
            self.constrain_op()
        return loss.item(), accuracy

    def evaluate(self, data_loader, split):
        self.model.eval()
        all_preds, all_labels = [], []

        # 获取整个数据集
        full_x = getattr(data_loader, f"{split}_x").to(self.device)
        full_y = getattr(data_loader, f"{split}_y").to(self.device)

        with torch.no_grad():
            # 前向传播
            outputs, _ = self.model(full_x)  # LSTM 输出

            # 展平输出和标签
            outputs, batch_y = self.flatten_outputs_and_labels(outputs, full_y)

            # 计算 loss 和 accuracy
            loss, accuracy = self.compute_loss_and_accuracy(outputs, batch_y)

            if split == "test":
                all_preds = outputs.argmax(dim=1).cpu().numpy()
                all_labels = batch_y.cpu().numpy()

        # 打印分类报告（仅在测试集上）
        if split == "test":
            print(classification_report(all_labels, all_preds))
        # 更新学习率
        return accuracy , loss.item()

    def save_model(self):
        """
        保存模型到指定路径。
        """
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """
        从指定路径加载模型。
        """
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print(f"Model loaded from {self.model_path}")

    def build_state(self, init_state,batch_size, hidden_size, device):
        model_type = self.model_type
        if init_state is not None:
            if model_type == 'lstm':
                hx = (init_state[0].unsqueeze(0), init_state[1].unsqueeze(0)) if not self.time_major else init_state
                return hx
            else:
                return init_state
        if model_type == 'lstm':
            return self.build_lstm_state(batch_size, hidden_size, device)
        else:
            return self.build_ltc_state(batch_size, hidden_size, device)

    def build_lstm_state(self, batch_size, hidden_size,device):
        # 确保 LSTM 初始状态是 (h_t, c_t) 形式的元组
        h_0 = torch.zeros(batch_size, hidden_size, device=device)
        c_0 = torch.zeros(batch_size, hidden_size, device=device)
        return (h_0, c_0)  # 返回元组 (h_t, c_t)

    def build_ltc_state(self, batch_size, hidden_size,device):
        hx = torch.zeros(batch_size, hidden_size, device=device)
        return hx
