import argparse

from liquid_dataset.ozone import OzoneData
from experiments_retest.test_model import  TestModel


if __name__ == "__main__":

    # https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--sparsity',default=0.0,type=float)

    args = parser.parse_args()

    path = '/mnt/f/dataset/liquid_data/ozone/eighthr.data'

    ozone_data = OzoneData(path)
    model = TestModel(model_type = args.model,input_size=72,model_size=args.size,sparsity_level=args.sparsity,output_size=2)

    model.fit(ozone_data,epochs=args.epochs,log_period=args.log)
#
#
# Training the OzoneModel with Gesture Data
# def train_gesture_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     path ='/mnt/f/dataset/liquid_data/gesture'
#     gesture_data = GestureData(path = path)
#
#     model = TestModel(model_type="ltc", input_size=32,model_size=32,output_size=5).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#
#     # Training Loop
#     epochs = 10
#     for epoch in range(epochs):
#         model.train()
#         total_loss, total_acc = 0, 0
#         for batch_x, batch_y in gesture_data.iterate_train():
#             # 将 NumPy 数组转换为 PyTorch 张量，并移动到正确的设备 (CPU 或 GPU)
#             batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
#             batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
#
#             optimizer.zero_grad()
#
#             # 获取模型输出，并取最后一个时间步的预测结果
#             outputs = model(batch_x)  # [batch_size, seq_len, num_classes]
#
#             # 计算损失
#             loss = criterion(outputs, batch_y[:, -1])  # [batch_size] 标签为1D
#             print(f"当前损失: {loss.item()}")
#
#             # 反向传播并更新参数
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             total_acc += (outputs.argmax(dim=1) == batch_y[:, -1]).float().mean().item()
#
#         print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {total_acc / len(gesture_data.train_x):.4f}")
#
#     print("Training complete.")
#
#
# if __name__ == "__main__":
#     train_gesture_model()