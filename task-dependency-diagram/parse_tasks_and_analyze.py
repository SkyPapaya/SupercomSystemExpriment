import networkx as nx

def load_tasks_from_file(file_path):
    G = nx.DiGraph()
    durations = {}

    with open(file_path, 'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # 忽略空行或注释

            parts = line.split(',')
            task = parts[0].strip()
            time = int(parts[1].strip())
            durations[task] = time

            if len(parts) > 2 and parts[2].strip():
                dependencies = parts[2].strip().split('|')
                for dep in dependencies:
                    dep = dep.strip()
                    G.add_edge(dep, task, weight=durations[task])
            else:
                G.add_node(task)

    return G, durations

def analyze_critical_path(G):
    path = nx.dag_longest_path(G, weight='weight')
    total_time = nx.dag_longest_path_length(G, weight='weight')
    return path, total_time

if __name__ == "__main__":
    file_path = "tasks.txt"
    G, durations = load_tasks_from_file(file_path)

    path, total_time = analyze_critical_path(G)

    print("\n===  关键路径分析 ===")
    print("关键路径:", " → ".join(path))
    print("估算总执行时间:", total_time, "秒")

    print("\n===  高优先级资源建议 ===")
    for task in path:
        print(f"为 {task} 分配 高性能资源（如GPU/高速缓存/优先核）")
