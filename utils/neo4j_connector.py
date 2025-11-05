import torch
from neo4j import GraphDatabase
from typing import List, Dict, Any, Tuple
import time
import json
import pyahocorasick


class Neo4jConnector:
    """
    用于连接 Neo4j、执行实体链接和动态获取子图的类。
    """

    def __init__(self, uri, user, password, grounding_pattern_path):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Initializing Neo4j Connector...")

        print(f"Loading grounding vocabulary from {grounding_vocab_path}...")
        self.automaton = pyahocorasick.Automaton()
        try:
            with open(grounding_vocab_path, "r", encoding="utf8") as fin:
                for idx, line in enumerate(fin):
                    entity_name = line.strip()
                    if entity_name:
                        # (重要)
                        # Aho-Corasick 允许我们添加一个 (key, value) 对
                        # 我们把实体名称 (key) 和它自己 (value) 关联起来
                        self.automaton.add_word(entity_name, entity_name)

            print(f"Loaded {idx + 1} entities into Aho-Corasick automaton.")

            # 构建自动机，使其可用于搜索
            self.automaton.make_automaton()
            print("Grounding automaton built.")

        except FileNotFoundError:
            print(f"错误：实体词汇表文件未找到: {grounding_vocab_path}")
            print("请先运行 utils/build_neo4j_vocab.py 来生成该文件。")
            raise
        except Exception as e:
            print(f"加载自动机时出错: {e}")
            raise

        def close(self):
            self.driver.close()

        def link_entities(self, query: str) -> List[str]:


            mentioned_concepts = set()

            # self.automaton.iter(query)
            # 会在 query 中查找自动机中的所有(中文)实体
            # 它返回一个 (end_index, value) 的元组
            # value 就是我们_添加_的实体名称

            for end_index, original_concept in self.automaton.iter(query):
                # original_concept 已经是 "苹果公司" 或 "史蒂夫 乔布斯"
                mentioned_concepts.add(original_concept)

            return list(mentioned_concepts)

        def fetch_subgraph_for_entities(self, entities: List[str], hops: int = 1) -> Dict[str, Any]:
            """
            (已修改 - 简化)
            为一组种子实体动态获取一个 n-hop 子图。
            """
            if not entities:
                return {'nodes': [], 'edges': []}

            # (重要)
            # `entities` 列表现在是 ["苹果公司", "史蒂夫 乔布斯"]
            # 这正是 Neo4j 中 'name' 属性的值。

            query = f"""
            MATCH (seed)
            WHERE seed.name IN $entities // (!! 直接使用 $entities !!)
            // 获取 1-hop 邻居和连接它们的边
            CALL {{
                WITH seed
                MATCH (seed)-[r]-(neighbor)
                RETURN seed, r, neighbor
            }}
            RETURN seed, r, neighbor
            """

            nodes = {}  # 使用字典去重
            edges = []

            with self.driver.session() as session:
                # (!! 修改了变量名 !!)
                results = session.run(query, entities=entities, hops=hops)

                for record in results:
                    seed_node = record["seed"]
                    rel = record["r"]
                    neighbor_node = record["neighbor"]


                    # 处理节点 (包含属性和类型)
                    for node in [seed_node, neighbor_node]:
                        node_id = node.element_id
                        if node_id not in nodes:
                            nodes[node_id] = {
                                'id': node.element_id,
                                'labels': list(node.labels),  # 英文
                                'properties': dict(node)  # 包含 'name' (中文)
                            }

                    # 处理边 (包含属性和类型)
                    edges.append({
                        'id': rel.element_id,
                        'type': rel.type,  # 英文
                        'start_node_id': rel.start_node.element_id,
                        'end_node_id': rel.end_node.element_id,
                        'properties': dict(rel)
                    })

            return {'nodes': list(nodes.values()), 'edges': edges}



