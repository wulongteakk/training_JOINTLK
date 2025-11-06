import json
import argparse
from neo4j import GraphDatabase
from tqdm import tqdm


def fetch_neo4j_entity_labels(driver, cypher_query):
    """

    获取neo4j节点标签映射ID
    """
    print("Connecting to Neo4j to fetch entity labels...")
    all_labels = set()

    with driver.session() as session:
        results = session.run(cypher_query)
        for record in tqdm(results, desc="Fetching entities"):
            # 从结果中获取标签列表（labels(n)返回的是列表，如["Person", "Organization"]）
            labels = record.get("labels", [])
            # 遍历列表中的每个标签，过滤无效值后添加到集合
            if isinstance(labels, list):
                for label in labels:
                    # 只保留非空字符串标签（去除首尾空格）
                    if isinstance(label, str) and label.strip():
                        cleaned_label = label.strip()
                        all_labels.add(cleaned_label)

    # 将去重后的标签转为有序列表（sorted保证顺序固定，索引一致）
    sorted_labels = sorted(all_labels)  # 按字母排序，若无需排序可直接用 list(all_labels)
    # 生成 {标签: 索引} 字典（enumerate从0开始生成索引）
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    print(label_to_index)
    print(f"Fetched {len(label_to_index)} unique entity labels from Neo4j.")
    return label_to_index


def fetch_neo4j_relation_mapping(driver, cypher_query):
    """
    获取Neo4j中关系类型的映射字典 {关系类型: 索引}（索引从0开始）
    """
    print("Connecting to Neo4j to fetch relationship types...")
    all_relations = set()  # 用集合自动去重关系类型

    with driver.session() as session:
        results = session.run(cypher_query)
        for record in tqdm(results, desc="Fetching relationships"):
            # 从结果中获取关系类型（对应查询中的 rel_type 字段）
            rel_type = record.get("rel_type")  # 关键：提取关系类型

            # 过滤无效的关系类型（非字符串、空值、纯空格）
            if isinstance(rel_type, str) and rel_type.strip():
                cleaned_rel = rel_type.strip()
                all_relations.add(cleaned_rel)  # 添加到集合去重

    # 将去重后的关系类型转为有序列表（保证索引固定）
    sorted_relations = sorted(all_relations)  # 按字母排序，可选
    # 生成 {关系类型: 索引} 字典
    relation_to_index = {rel: idx for idx, rel in enumerate(sorted_relations)}

    print(f"Fetched {len(relation_to_index)} unique relationship types from Neo4j.")
    return relation_to_index

def generate_reflection_from_neo4j(neo4j_config, cypher_query,query_edges, output_path_1, output_path_2):
    """
    主函数：连接 Neo4j，获取实体，保存为纯文本词汇表。
    """


    try:
        driver = GraphDatabase.driver(neo4j_config['uri'], auth=(neo4j_config['user'], neo4j_config['password']))
        driver.verify_connectivity()
        print("Neo4j connection successful.")
    except Exception as e:
        print(f"Failed to connect to Neo4j at {neo4j_config['uri']}: {e}")
        return


    node_type_mapping = fetch_neo4j_entity_labels(driver, cypher_query)
    edge_type_mapping = fetch_neo4j_relation_mapping(driver, query_edges)
    driver.close()


    # 保存节点标签映射（使用json.dump将字典转为JSON）
    print(f"Saving {len(node_type_mapping)} node types to {output_path_1}...")
    with open(output_path_1, "w", encoding="utf8") as fout:
        json.dump(node_type_mapping, fout, ensure_ascii=False, indent=2)

    print(f"Successfully saved node types reflection to {output_path_1}")

    # 保存关系类型映射
    print(f"Saving {len(edge_type_mapping)} edge types to {output_path_2}...")
    with open(output_path_2, "w", encoding="utf8") as fout:
        json.dump(edge_type_mapping, fout, ensure_ascii=False, indent=2)  # 关键修改

    print(f"Successfully saved edge types reflection to {output_path_2}")


if __name__ == "__main__":


    NEO4J_CONFIG = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "123456789"
    }

    query_entities = """
    MATCH (n)
    WHERE NOT n:Chunk AND NOT n:Document 
    RETURN DISTINCT labels(n) AS labels
    """

    query_edges = """
    MATCH (h)-[r]->(t) 
    WHERE NOT h:Chunk AND NOT h:Document AND NOT t:Chunk AND NOT t:Document 
    RETURN h.id AS head, t.id AS tail, type(r) AS rel_type
    """


    OUTPUT_PATH_1 ="D:\\JointLK_1\\data\\node_types.json"
    OUTPUT_PATH_2 = "D:\\JointLK_1\\data\\edge_types.json"


    # --- 运行 ---
    generate_reflection_from_neo4j(NEO4J_CONFIG, query_entities, query_edges,OUTPUT_PATH_1,OUTPUT_PATH_2)
