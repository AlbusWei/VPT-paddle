import paddle
from ResNet import ResNet50
from SwinT import swin_tiny
from dataPretreatment import generate_dataloader, preview
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy

def resnet_train(train_loader,valid_loader,save_dir = '/home/aistudio/ck_resnet',callback = None):
    BATCH_SIZE = 512
    EPOCHS = 300  # 训练次数
    # decay_steps = int(len(trn_dateset) / BATCH_SIZE * EPOCHS)

    model = paddle.Model(ResNet50(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.1,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    # model.load('ck_resnet/100.pdparams')

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy())
    # 启动训练
    model.fit(train_loader,
              valid_loader,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              eval_freq=5,  # 多少epoch 进行验证
              save_freq=10,  # 多少epoch 进行模型保存
              log_freq=100,  # 多少steps 打印训练信息
              #   save_dir='/home/aistudio/checkpoint',
              save_dir=save_dir,
              callbacks=callback)
    return model

def swinT_train(train_loader,valid_loader,save_dir = '/home/aistudio/checkpoint',callback = None):
    BATCH_SIZE = 12
    EPOCHS = 10  # 训练次数
    decay_steps = int(train_loader.len / BATCH_SIZE * EPOCHS)

    model = paddle.Model(swin_tiny(num_classes=2))
    base_lr = 0.0125
    lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=decay_steps, end_lr=0.0)
    # 定义优化器
    optimizer = paddle.optimizer.Momentum(learning_rate=lr,
                                          momentum=0.9,
                                          weight_decay=L2Decay(1e-4),
                                          parameters=model.parameters())

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
    # 启动训练
    model.fit(train_loader,
              valid_loader,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              eval_freq=5,  # 多少epoch 进行验证
              save_freq=5,  # 多少epoch 进行模型保存
              log_freq=100,  # 多少steps 打印训练信息
              save_dir=save_dir,
              callbacks=callback)
    return model


def main():

    # 文件地址
    train_txt = "work/train_list.txt"
    test_txt = "work/test_list.txt"
    val_txt = "work/val_list.txt"

    train_loader, valid_loader, test_loader = generate_dataloader(BATCH_SIZE = 512)
    preview(train_loader)

    resnet_train(train_loader,valid_loader)



if __name__ == "__main__":
    main()
