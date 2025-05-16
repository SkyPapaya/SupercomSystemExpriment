
import networkx as nx

def schedule_with_priority(task_graph, nodes, predictor):
    sorted_tasks = sorted(task_graph.nodes, key=lambda t: task_graph.nodes[t]["task"].priority, reverse=True)
    current_time = 0
    for task_name in sorted_tasks:
        task = task_graph.nodes[task_name]["task"]
        predictor(task)

        best_node = None
        for node in nodes:
            if node.can_run(task):
                best_node = node
                break

        if best_node:
            task.start_time = current_time
            task.end_time = current_time + task.predicted_runtime
            best_node.assign(task)
            print(f"[已调度] {task.name} -> {best_node.name} 运行 {task.predicted_runtime:.2f} 秒 "
                  f"[{task.start_time:.2f}s ~ {task.end_time:.2f}s]")
            current_time = task.end_time  # 串行模拟，可改为并行时间轴
        else:
            print(f"[调度失败] 任务 {task.name} 分配失败。")
