class Task:
    def __init__(self, name, task_type, cpu_demand, gpu_demand, io_demand, priority=1):
        self.name = name
        self.task_type = task_type
        self.cpu_demand = cpu_demand
        self.gpu_demand = gpu_demand
        self.io_demand = io_demand
        self.priority = priority
        self.predicted_runtime = None  # 会由 predictor 设置
        self.start_time = None
        self.end_time = None
