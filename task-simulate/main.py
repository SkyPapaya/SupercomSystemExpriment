import matplotlib.pyplot as plt
import networkx as nx
from task import Task
from node import Node
from predictor import predict_execution_time
from priority_scheduler import schedule_with_priority
from optimizer import rebalance
from migration import migrate_tasks
from communication import simulate_communication

def create_tasks():
    return [
        Task("图像识别", "视觉推理", 4, 8, 200, priority=3),
        Task("传感融合", "数据预处理", 6, 0, 300, priority=2),
        Task("路径重规划", "计算控制", 10, 2, 100, priority=2),
        Task("热成像处理", "图像增强", 5, 6, 250, priority=1),
        Task("飞行日志写入", "IO存储", 2, 0, 500, priority=1),
        Task("异常检测", "AI分析", 8, 10, 150, priority=3),
        Task("位置同步", "通信计算", 3, 0, 100, priority=1),
        Task("多机调度", "控制决策", 7, 4, 200, priority=2)
    ]

def create_nodes():
    return [
        Node("母机-A", cpu_cores=32, gpu_mem=32, io_bw=1000, is_mother=True),
        Node("子机-1", cpu_cores=8, gpu_mem=4, io_bw=100),
        Node("子机-2", cpu_cores=8, gpu_mem=4, io_bw=100),
        Node("子机-3", cpu_cores=8, gpu_mem=4, io_bw=100),
    ]

def build_task_graph(tasks):
    G = nx.DiGraph()
    for task in tasks:
        G.add_node(task.name, task=task)
    for i in range(len(tasks) - 1):
        if i % 2 == 0:
            G.add_edge(tasks[i].name, tasks[i + 1].name)
    return G

def show_task_graph(G):
    pos = nx.spring_layout(G)
    labels = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightgreen")
    plt.title("任务依赖图 AOV")
    plt.show()

def show_summary(nodes):
    print("\n--- 节点任务总结 ---")
    for node in nodes:
        s = node.summary()
        print(f"{s['name']} ({'母机' if s['is_mother'] else '子机'})")
        print(f"  任务: {s['tasks']}")
        print(f"  负载: CPU={s['load']['cpu']} GPU={s['load']['gpu']} IO={s['load']['io']}")

def run():
    tasks = create_tasks()
    nodes = create_nodes()
    graph = build_task_graph(tasks)
    schedule_with_priority(graph, nodes, predict_execution_time)
    rebalance(nodes)
    migrate_tasks(nodes)
    simulate_communication(nodes)
    show_summary(nodes)
    #show_task_graph(graph)

if __name__ == "__main__":
    run()
