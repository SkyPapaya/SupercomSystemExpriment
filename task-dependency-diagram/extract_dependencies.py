import ast

def extract_functions_and_calls_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    tree = ast.parse(code)
    dependencies = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fname = node.name
            dependencies[fname] = []

            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    dependencies[fname].append(n.func.id)

    return dependencies

def write_dependency_file(dependencies, output_path='tasks.txt'):
    with open(output_path, 'w', encoding='utf-8') as f:
        for func, deps in dependencies.items():
            line = f"{func}, 1, {'|'.join(deps) if deps else ''}\n"
            f.write(line)
    print(f"✅ 依赖关系写入文件：{output_path}")

if __name__ == "__main__":
    file_path = "parse_tasks_and_analyze.py"  # 替换成你的脚本路径
    deps = extract_functions_and_calls_from_file(file_path)
    write_dependency_file(deps)
