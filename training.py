import paddle
from ResNet import ResNet50
from SwinT import swin_tiny
from CSwin import CSWinTransformer_tiny_224
from ViT import ViT_small_patch16_224
from DeiT import DeiT_tiny_patch16_224
from dataPretreatment import generate_dataloader
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy, Auc


def resnet_train(train_loader, valid_loader, save_dir='./checkpoint/ResNet50', callback=None):
    BATCH_SIZE = 512
    EPOCHS = 200  # 训练次数

    model = paddle.Model(ResNet50(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    model.load('pretrain/ResNet50_pretrained.pdparams')

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy())
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


def swinT_train(train_loader, valid_loader, save_dir='./checkpoint/SwinT', callback=None):
    BATCH_SIZE = 256
    EPOCHS = 200  # 训练次数

    model = paddle.Model(swin_tiny(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    model.load('pretrain/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams', skip_mismatch=True)

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy())
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


def ViT_train(train_loader, valid_loader, save_dir='./checkpoint/ViT', callback=None):
    BATCH_SIZE = 512
    EPOCHS = 200  # 训练次数

    model = paddle.Model(ViT_small_patch16_224(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    # 加载预训练权重
    model.load('pretrain/ViT_small_patch16_224_pretrained.pdparams', skip_mismatch=True)

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy())
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


def CSwin_train(train_loader, valid_loader, save_dir='./checkpoint/CSwin', callback=None):
    BATCH_SIZE = 512
    EPOCHS = 200  # 训练次数

    model = paddle.Model(CSWinTransformer_tiny_224(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    # 加载预训练权重
    model.load('pretrain/CSWinTransformer_tiny_224_pretrained.pdparams', skip_mismatch=True)

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy())
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


def DeiT_train(train_loader, valid_loader, save_dir='./checkpoint/DeiT', callback=None):
    BATCH_SIZE = 512
    EPOCHS = 200  # 训练次数

    model = paddle.Model(DeiT_tiny_patch16_224(num_classes=2))
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.99], dtype="float32")

    optimizer = paddle.optimizer.AdamW(learning_rate=0.0001,
                                       parameters=model.parameters(),
                                       beta1=beta1,
                                       beta2=beta2,
                                       weight_decay=0.01)

    # 加载预训练权重
    model.load('pretrain/DeiT_tiny_patch16_224_pretrained.pdparams', skip_mismatch=True)

    model.prepare(optimizer, CrossEntropyLoss(), Accuracy())
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

    callback = paddle.callbacks.VisualDL(log_dir='./log/')

    train_loader, valid_loader, test_loader = generate_dataloader()
    # preview(train_loader)

    # model = resnet_train(train_loader, valid_loader,callback)
    model = ViT_train(train_loader, valid_loader, callback=callback)
    # model = swinT_train(train_loader, valid_loader,callback)
    # model = CSwin_train(train_loader, valid_loader,callback=callback)
    # model = DeiT_train(train_loader, valid_loader,callback=callback)

    # 测试
    model.evaluate(test_loader, log_freq=30, verbose=2)


if __name__ == "__main__":
    main()
