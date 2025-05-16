
def predict_execution_time(task):
    # 简单估算函数，根据资源需求粗略估算执行时间
    return round(0.1 * task.cpu_req + 0.2 * task.gpu_req + 0.05 * task.io_req, 2)
