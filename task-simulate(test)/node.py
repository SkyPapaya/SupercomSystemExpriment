
class Node:
    def __init__(self, name, cpu_cores, gpu_mem, io_bw, is_mother=False):
        self.name = name
        self.cpu_cores = cpu_cores
        self.gpu_mem = gpu_mem
        self.io_bw = io_bw
        self.is_mother = is_mother
        self.load = {"cpu": 0, "gpu": 0, "io": 0}
        self.tasks = []

    def can_run(self, task):
        return (self.cpu_cores - self.load["cpu"] >= task.cpu_req and
                self.gpu_mem - self.load["gpu"] >= task.gpu_req and
                self.io_bw - self.load["io"] >= task.io_req)

    def assign(self, task):
        self.load['cpu'] += task.cpu_req
        self.load['gpu'] += task.gpu_req
        self.load['io'] += task.io_req
        self.tasks.append(task.name)
        print(f"[调度] 任务 {task.name} 分配给 {self.name}")

    def summary(self):
        return {
            "name": self.name,
            "is_mother": self.is_mother,
            "load": self.load,
            "tasks": self.tasks
        }
