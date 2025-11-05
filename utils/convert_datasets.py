import json
import os

# --- 配置 ---
INPUT_FILE = "D:\JointLK\JointLK\data\datasets-sharegpt-2025-03-26.json"
OUTPUT_FILE = "D:\JointLK\JointLK\data\construction_qa_intermediate.jsonl"


# ------------

def convert_sharegpt_to_jsonl(input_path, output_path):
    """
    读取 ShareGPT 格式的 JSON 文件，提取 "user" (问题) 和 "assistant" (答案)
    内容，并将其转换为 .jsonl 文件 (每行一个JSON对象)。
    """

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件 '{input_path}' 未找到。")
        print("请确保 'datasets-sharegpt-2025-03-26.json' 文件与此脚本在同一目录中。")
        return

    print(f"正在读取输入文件: {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：解析 JSON 文件失败。{e}")
        return
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return

    processed_count = 0
    print(f"正在将数据写入输出文件: {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, entry in enumerate(data):
            messages = entry.get('messages', [])

            # 确保数据格式是 [user, assistant]
            if len(messages) >= 2 and \
                    messages[0].get('role') == 'user' and \
                    messages[1].get('role') == 'assistant':

                question = messages[0].get('content')
                answer_raw = messages[1].get('content', '')

                # 移除 <think>...</think> 块（如果存在）
                think_start = answer_raw.find('<think>')
                think_end = answer_raw.find('</think>')
                if think_start != -1 and think_end != -1:
                    answer = answer_raw[think_end + len('</think>'):].strip()
                else:
                    answer = answer_raw.strip()

                if question and answer:
                    new_entry = {
                        "id": f"construction_qa_{i + 1:04d}",
                        "question": question,
                        # "answer": answer

                    }

                    # 写入 .jsonl 文件
                    f_out.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                    processed_count += 1

    print(f"处理完成。总共 {processed_count} 条数据已保存到 {output_path}")


if __name__ == "__main__":
    convert_sharegpt_to_jsonl(INPUT_FILE, OUTPUT_FILE)