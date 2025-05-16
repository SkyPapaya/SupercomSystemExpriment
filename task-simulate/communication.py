import random
import time

def simulate_communication(nodes):
    print("\n[通信] 模拟母机与子机间通信...")
    mother = next((n for n in nodes if n.is_mother), None)
    if not mother:
        print("[通信] 无母机，通信跳过")
        return
    for node in nodes:
        if not node.is_mother:
            latency = round(random.uniform(0.5, 2.0), 2)
            print(f"[通信] {node.name} → {mother.name} 延迟 {latency}s")
            time.sleep(0.1)
