def migrate_tasks(nodes):
    print("\n[迁移] 检查节点间任务迁移机会...")
    overloaded = [n for n in nodes if n.load['cpu'] > 20]
    underloaded = [n for n in nodes if n.load['cpu'] < 10]
    for o_node in overloaded:
        for task_name in list(o_node.tasks):
            for u_node in underloaded:
                if task_name not in u_node.tasks:
                    print(f"[迁移] 将任务 {task_name} 从 {o_node.name} 迁移到 {u_node.name}")
                    o_node.tasks.remove(task_name)
                    u_node.tasks.append(task_name)
                    break
