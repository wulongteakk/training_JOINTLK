import json
import argparse
from neo4j import GraphDatabase
from tqdm import tqdm


def fetch_neo4j_entity_names(driver, cypher_query):
    """
    (已修改)
    从 Neo4j 数据库获取所有实体的 'name' 属性。
    """
    print("Connecting to Neo4j to fetch entity names...")
    entity_names = set()
    with driver.session() as session:
        results = session.run(cypher_query)
        for record in tqdm(results, desc="Fetching entities"):
            entity_name = record["name"]
            # 确保是有效的、非空的(中文)字符串
            if entity_name and isinstance(entity_name, str) and entity_name.strip():
                # (重要) 确保实体名称不包含换行符，并去除首尾空格
                entity_names.add(entity_name.strip())

    print(f"Fetched {len(entity_names)} unique entity names from Neo4j.")
    return list(entity_names)


def generate_vocab_from_neo4j(neo4j_config, cypher_query, output_path):
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


    entity_names_from_db = fetch_neo4j_entity_names(driver, cypher_query)
    driver.close()


    print(f"Saving {len(entity_names_from_db)} entities to {output_path}...")
    with open(output_path, "w", encoding="utf8") as fout:
        for name in entity_names_from_db:
            fout.write(name + '\n')

    print(f"Successfully saved entity vocabulary to {output_path}")


if __name__ == "__main__":


    NEO4J_CONFIG = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "123456789"
    }

    CYPHER_QUERY = """
    MATCH (n)
    WHERE NOT n:Chunk AND NOT n:Document 
    RETURN DISTINCT n.id AS name
    """


    OUTPUT_PATH = "D:\\JointLK\\JointLK\\data\\entity_vocab.txt"

    # --- 运行 ---
    generate_vocab_from_neo4j(NEO4J_CONFIG, CYPHER_QUERY, OUTPUT_PATH)









