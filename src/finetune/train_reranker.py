"""
Reranker LoRA 微调模块
使用 sentence-transformers + PEFT (LoRA) 微调 BGE-Reranker
"""
import json
import os
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample


class RerankerTrainer:
    """Reranker 微调器"""

    def __init__(
        self,
        base_model: str = "BAAI/bge-reranker-v2-m3",
        output_dir: str = "./models/reranker-finetuned",
        device: str = None,
    ):
        """
        Args:
            base_model: 基座 Reranker 模型
            output_dir: 微调后模型保存路径
            device: 训练设备，None 为自动检测
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"使用设备: {self.device}")

    def prepare_data(self, data_path: str) -> List[InputExample]:
        """
        将微调数据转换为 InputExample 格式

        输入格式 (JSON):
        [
            {
                "query": "云服务器怎么扩容",
                "positive": "本文介绍如何扩容云盘...",
                "negative": ["CDN加速需要备案...", "OSS按量计费..."]
            },
            ...
        ]

        输出：每条记录生成 1 个正例 + len(negative) 个负例
        """
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for item in raw_data:
            query = item["query"]

            # 正例 (label=1)
            samples.append(InputExample(
                texts=[query, item["positive"]],
                label=1,
            ))

            # 负例 (label=0)
            for neg_text in item.get("negative", []):
                samples.append(InputExample(
                    texts=[query, neg_text],
                    label=0,
                ))

        print(f"构造训练样本: {len(samples)} 条 "
              f"(正例: {len(raw_data)}, 负例: {len(samples) - len(raw_data)})")

        return samples

    def train(
        self,
        train_data_path: str,
        eval_data_path: str = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        max_length: int = 512,
        save_best: bool = True,
    ):
        """
        训练 Reranker

        Args:
            train_data_path: 训练数据 JSON 路径
            eval_data_path: 验证数据 JSON 路径（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            warmup_steps: 预热步数
            use_lora: 是否使用 LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            max_length: 最大输入长度
            save_best: 是否只保存最优模型
        """
        # 准备数据
        train_samples = self.prepare_data(train_data_path)
        train_dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=batch_size,
        )

        eval_dataloader = None
        if eval_data_path and os.path.exists(eval_data_path):
            eval_samples = self.prepare_data(eval_data_path)
            eval_dataloader = DataLoader(
                eval_samples,
                shuffle=False,
                batch_size=batch_size,
            )

        # 加载基座模型
        print(f"加载基座模型: {self.base_model}")
        model = CrossEncoder(
            self.base_model,
            max_length=max_length,
            device=self.device,
        )

        # LoRA 微调
        if use_lora:
            model = self._apply_lora(
                model,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        # 训练参数
        print(f"\n开始训练...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  LoRA: {'启用 (r=' + str(lora_r) + ', alpha=' + str(lora_alpha) + ')' if use_lora else '禁用'}")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.output_dir}\n")

        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=str(self.output_dir),
            show_progress_bar=True,
        )

        print(f"\n训练完成！模型已保存到 {self.output_dir}")

    def _apply_lora(
        self,
        model: CrossEncoder,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
    ) -> CrossEncoder:
        """对 CrossEncoder 内部的 Transformer 应用 LoRA"""
        from peft import get_peft_model, LoraConfig, TaskType

        # CrossEncoder.model 是底层的 Transformer 模型
        transformer = model.model

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["query", "value", "key", "dense"],  # BERT 类 attention 层
            bias="none",
        )

        try:
            model.model = get_peft_model(transformer, lora_config)

            # 统计可训练参数
            trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.model.parameters())
            print(f"LoRA 已应用: 可训练参数 {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        except Exception as e:
            print(f"LoRA 应用失败: {e}")
            print("将使用全量微调（需要更多显存）")

        return model


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="微调 BGE-Reranker")
    parser.add_argument("--train_data", default="data/eval/reranker_train_data.json",
                        help="训练数据路径")
    parser.add_argument("--eval_data", default="data/eval/reranker_eval_data.json",
                        help="验证数据路径")
    parser.add_argument("--base_model", default="BAAI/bge-reranker-v2-m3",
                        help="基座模型名称")
    parser.add_argument("--output_dir", default="./models/reranker-finetuned",
                        help="输出目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--no_lora", action="store_true",
                        help="禁用 LoRA，使用全量微调")

    args = parser.parse_args()

    trainer = RerankerTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
    )

    trainer.train(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
    )


if __name__ == "__main__":
    main()
