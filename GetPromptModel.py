import paddle
from ViTmodel.CSwin import CSWinTransformer_base_224
from ViTmodel.ViT import ViT_base_patch16_224
from ViTmodel.SwinT import SwinTransformer_base_patch4_window7_224
from structure import *

model_dict = {
    "ViT": {
        "function": ViT_base_patch16_224,
        "save_dir": './checkpoint/ViT',
        "pretrain": 'pretrain/ViT_base_patch16_224_pretrained.pdparams'
    },
    "SwinT": {
        "function": SwinTransformer_base_patch4_window7_224,
        "save_dir": './checkpoint/SwinT',
        "pretrain": 'pretrain/SwinTransformer_base_patch4_window7_224_pretrained.pdparams'
    },
    "CSwinT": {
        "function": CSWinTransformer_base_224,
        "save_dir": './checkpoint/CSwin',
        "pretrain": 'pretrain/CSWinTransformer_base_224_pretrained.pdparams'
    }
}


def build_promptmodel(num_classes=2, edge_size=224, model_idx='ViT', patch_size=16,
                      Prompt_Token_num=10, VPT_type="Deep"):
    # VPT_type = "Deep" / "Shallow"

    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        # import timm
        # from pprint import pprint
        # model_names = timm.list_models('*vit*')
        # pprint(model_names)

        # basic_model = paddle.Model(model_dict[model_idx]["function"](class_num=num_classes))
        # basic_model.load(model_dict[model_idx]["pretrain"], skip_mismatch=True)

        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type)

        basic_model_state = paddle.load(model_dict[model_idx]["pretrain"])
        model.set_state_dict(basic_model_state, False)
        model.New_CLS_head(num_classes)
        model.Freeze()
    else:
        print("The model is not difined in the Prompt script！！")
        return -1

    try:
        img = paddle.randn((1, 3, edge_size, edge_size))
        preds = model.predict_batch([img])  # (1, class_number)
        print('test model output：', preds)
    except:
        print("Problem exist in the model defining process！！")
        return -1
    else:
        print('model is ready now!')
        return model
