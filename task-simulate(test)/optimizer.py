
def rebalance(nodes, threshold=0.7):
    print("\n[优化] 正在进行动态调度优化...")
    avg_cpu = sum(n.load["cpu"] for n in nodes) / len(nodes)
    for node in nodes:
        if node.load["cpu"] > threshold * avg_cpu:
            print(f"[提示] 节点 {node.name} CPU 负载偏高，可考虑迁移任务")
