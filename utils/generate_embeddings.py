import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import argparse
import os


def generate_embeddings(
        vocab_file_path: str,
        output_file_path: str,
        model_name: str,
        batch_size: int = 64,
        context_placeholder: str = "__CONTEXT__"
):
    """
    从词汇表文件生成 .npy 嵌入矩阵。
    """

    # 1. 加载 Sentence Transformer 模型
    print(f"正在加载模型: {model_name}...")
    # 自动使用 GPU (cuda) (如果可用)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    # 获取嵌入维度
    emb_dim = model.get_embedding_dimension()
    print(f"模型加载完毕。嵌入维度: {emb_dim}")

    # 2. 读取词汇表
    print(f"正在读取词汇表: {vocab_file_path}")
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        vocab_lines = [line.strip() for line in f.readlines()]

    num_entities = len(vocab_lines)
    print(f"词汇表中共找到 {num_entities} 个实体。")

    # 3. 准备文本和占位符
    texts_to_encode = []
    placeholder_indices = []

    for i, line in enumerate(vocab_lines):
        if line == context_placeholder:
            placeholder_indices.append(i)
        else:
            texts_to_encode.append(line)

    print(f"找到 {len(texts_to_encode)} 个真实实体需要编码。")
    print(f"找到 {len(placeholder_indices)} 个占位符 (例如: '{context_placeholder}')。")

    # 4. 批量编码真实实体 (GPU 加速)
    print("正在开始批量编码...")
    embeddings = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print("编码完成。")

    # 5. 构建最终的嵌入矩阵
    # 创建一个全零矩阵
    final_embedding_matrix = np.zeros((num_entities, emb_dim), dtype=np.float32)

    text_emb_iter = iter(embeddings)

    for i in range(num_entities):
        if i in placeholder_indices:
            # 占位符保持为零向量
            continue
        else:
            # 插入真实实体的嵌入
            final_embedding_matrix[i] = next(text_emb_iter)

    # 6. 保存到 .npy 文件
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(output_file_path, final_embedding_matrix)
    print(f"\n成功! 嵌入矩阵已保存到: {output_file_path}")
    print(f"矩阵形状: {final_embedding_matrix.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从词汇表文件生成嵌入 .npy 文件")

    parser.add_argument(
        "--vocab_file",
        type=str,
        required=True,
        help="输入的 entity_vocab.txt 文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出的 ent_emb.npy 文件路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="要使用的 SentenceTransformer 模型名称 (必须支持中文)"
    )
    parser.add_argument(
        "--context_placeholder",
        type=str,
        default="__CONTEXT__",
        help="词汇表中用于上下文节点的占位符字符串"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="编码时的批量大小"
    )

    args = parser.parse_args()

    # 推荐的中文模型:
    # 1. paraphrase-multilingual-mpnet-base-v2 (通用多语言)
    # 2. shibing624/text2vec-base-chinese (专门的中文模型)
    # 3. ng-team/cjk-bert-base (中日韩)
    # 我们默认使用 paraphrase-multilingual-mpnet-base-v2，因为它非常强大且通用。

    generate_embeddings(
        args.vocab_file,
        args.output_file,
        args.model_name,
        args.batch_size,
        args.context_placeholder
    )
