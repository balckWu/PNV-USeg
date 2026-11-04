# 第三方源码与许可证

本目录保留了原实验实际使用的第三方源码及来源说明，没有修改其许可证。相关代码位于 `baselines/original_tsv3/third_party/`，以及部分原生实验副本。

上游项目包括 TransUNet、Swin-Unet、LViT、SAM-Med2D；CLIPSeg通过原有依赖和适配代码使用。Iris-2D是本项目的二维适配实现，不是原作者的三维Iris代码。

可在原目录查看随源码保留的许可证，例如：

- `baselines/original_tsv3/third_party/TransUNet/LICENSE`
- `baselines/original_tsv3/third_party/LViT/LICENSE`
- `baselines/original_tsv3/third_party/SAM-Med2D/LICENSE`

其他上游的授权与引用信息沿用相应README和安装脚本中的来源链接。源目录未提供的许可证不在本次整理中代填。发布时应保持第三方版权/引用信息，不将所有代码统一标为新的项目许可证。

本次没有加入任何第三方预训练权重、训练数据或生成结果。
