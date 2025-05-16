import time
import random
import threading

# 任务和资源定义
tasks = [
    {'id': 1, 'stateful': False, 'priority': 1, 'status': 'running', 'assigned_node': 'node1'},
    {'id': 2, 'stateful': True, 'priority': 5, 'status': 'running', 'assigned_node': 'node2'},
    {'id': 3, 'stateful': False, 'priority': 3, 'status': 'waiting', 'assigned_node': None},
]

nodes = {
    'node1': {'gpu_util': 50, 'memory_used': 60, 'status': 'active'},
    'node2': {'gpu_util': 95, 'memory_used': 85, 'status': 'active'},  # 高负载
    'node3': {'gpu_util': 20, 'memory_used': 30, 'status': 'active'},
}

# 模拟采集GPU利用率等监控数据
def monitor_nodes():
    while True:
        for node_id, stats in nodes.items():
            # 模拟数据波动
            stats['gpu_util'] = min(100, max(0, stats['gpu_util'] + random.randint(-10, 10)))
            stats['memory_used'] = min(100, max(0, stats['memory_used'] + random.randint(-5, 5)))
        time.sleep(5)

# 任务迁移函数
def migrate_task(task, target_node):
    print(f"准备迁移任务 {task['id']} 从 {task['assigned_node']} 到 {target_node}")
    if task['stateful']:
        # 有状态任务，模拟Checkpoint保存和加载
        print(f"任务 {task['id']} 保存Checkpoint...")
        time.sleep(1)
        print(f"任务 {task['id']} 在新节点加载Checkpoint...")
        time.sleep(1)
    else:
        # 无状态任务，直接重启
        print(f"任务 {task['id']} 无状态，直接重启...")
        time.sleep(0.5)
    task['assigned_node'] = target_node
    print(f"任务 {task['id']} 迁移完成")

# 抢占调度函数
def preemptive_scheduling():
    while True:
        # 找出高优先级等待任务
        waiting_high_priority = [t for t in tasks if t['status'] == 'waiting']
        if not waiting_high_priority:
            time.sleep(5)
            continue
        waiting_high_priority.sort(key=lambda x: x['priority'], reverse=True)
        task_to_schedule = waiting_high_priority[0]

        # 找出低优先级运行任务以抢占
        running_tasks = [t for t in tasks if t['status'] == 'running']
        running_tasks.sort(key=lambda x: x['priority'])
        for low_task in running_tasks:
            if low_task['priority'] < task_to_schedule['priority']:
                print(f"抢占任务 {low_task['id']} 资源给高优先级任务 {task_to_schedule['id']}")
                # 模拟抢占
                low_task['status'] = 'waiting'
                task_to_schedule['status'] = 'running'
                task_to_schedule['assigned_node'] = low_task['assigned_node']
                low_task['assigned_node'] = None
                break
        time.sleep(5)

# 动态调整策略
def dynamic_adjustment():
    while True:
        for node_id, stats in nodes.items():
            if stats['gpu_util'] > 90:
                # 找出此节点上的任务迁移
                tasks_on_node = [t for t in tasks if t['assigned_node'] == node_id]
                for task in tasks_on_node:
                    # 找空闲节点
                    free_nodes = [n for n, s in nodes.items() if s['gpu_util'] < 70 and n != node_id]
                    if free_nodes:
                        migrate_task(task, free_nodes[0])
                        break
        time.sleep(10)

# 弹性资源池（云资源扩容模拟）
def elastic_resource_pool():
    while True:
        cluster_load = np.mean([stats['gpu_util'] for stats in nodes.values()])
        if cluster_load > 80:
            print("检测到集群负载高，申请弹性云资源扩容...")
            new_node_id = f'node{len(nodes)+1}'
            nodes[new_node_id] = {'gpu_util': 0, 'memory_used': 0, 'status': 'active'}
            print(f"新节点 {new_node_id} 已加入集群")
        time.sleep(30)

# 启动线程模拟监控和调度
if __name__ == "__main__":
    import numpy as np
    monitor_thread = threading.Thread(target=monitor_nodes, daemon=True)
    adjust_thread = threading.Thread(target=dynamic_adjustment, daemon=True)
    preempt_thread = threading.Thread(target=preemptive_scheduling, daemon=True)
    elastic_thread = threading.Thread(target=elastic_resource_pool, daemon=True)

    monitor_thread.start()
    adjust_thread.start()
    preempt_thread.start()
    elastic_thread.start()

    while True:
        time.sleep(1)
