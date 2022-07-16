import argparse
import os
import queue
import sys
import pyperclip


def get_source_filename(directory: str, header: str):
    headers_extensions = [".hpp", ".h"]

    for ext in headers_extensions:
        if header.endswith(ext):
            source_filename = header.removesuffix(ext) + ".cpp"
            source_path = os.path.join(directory, source_filename)

            if os.path.exists(source_path):
                return source_filename

    return None


def find_library_file_path(include_file: str) -> str | None:
    if "/" not in include_file:
        return None

    print(f"Finding include {include_file}")

    library_name, file_name = include_file.split("/", 1)
    print(f"Library {library_name} File {file_name}")

    main_library_path = "wroclaw_zero/src"

    library_path = os.path.join(main_library_path, library_name)
    if not os.path.exists(library_path):
        return None

    file_path = os.path.join(library_path, "src", library_name, file_name)
    print(f"File path {file_path}")

    if os.path.exists(file_path):
        print(f"Found")
        return file_path

    return None


def generate_graph_and_codes(args) -> tuple[dict[str, set], dict[str, set]]:
    graph: dict[str, set] = {}
    codes: dict[str, str] = {}

    def parse_includes(directory: str, filename: str):
        file_path = os.path.join(directory, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        code_lines = []

        def is_local_include(line) -> bool:
            if line.startswith("#pragma once"):
                return True

            if not line.startswith("#include"):
                return False

            line = line.removeprefix("#include").lstrip().rstrip()

            if line.startswith("\""):
                # local include
                include_file = line.removeprefix("\"").removesuffix("\"")
                print(f"Local include = {include_file}")
                graph[file_path].add(os.path.join(directory, include_file))
                return True
            elif line.startswith("<"):
                include_file = line.removeprefix("<").removesuffix(">")
                include_path = find_library_file_path(include_file)

                if include_path is not None:
                    print(f"{include_file = } {include_path = }")
                    graph[file_path].add(include_path)
                    return True

            return False

        for line in lines:
            if not is_local_include(line):
                code_lines.append(line.lstrip())

        codes[file_path] = "".join(code_lines)

    def parse_file(file_path: str):
        if file_path in graph:
            return

        directory, filename = file_path.rsplit("/", 1)
        print(f"Parsing file {filename} in directory {directory}")

        graph[file_path] = set()

        source_filename = get_source_filename(directory, filename)

        if source_filename is not None:
            source_path = os.path.join(directory, source_filename)

            graph[source_path] = set()

            parse_includes(directory, source_filename)

            for include_file in graph[source_path]:
                parse_file(include_file)

        parse_includes(directory, filename)

        for include_path in graph[file_path]:
            parse_file(include_path)

    parse_file(args.main)

    return graph, codes


def generate_order(graph: dict[str, set]) -> list[str]:
    graphR = {u: set() for u in graph.keys()}
    in_deg = {u: 0 for u in graph.keys()}

    for u in graph.keys():
        for v in graph[u]:
            graphR[v].add(u)
            in_deg[u] += 1

    order = []

    q = queue.Queue()

    for u in in_deg.keys():
        if in_deg[u] == 0:
            q.put(u)

    while not q.empty():
        u = q.get()
        order.append(u)

        for v in graphR[u]:
            in_deg[v] -= 1

            if in_deg[v] == 0:
                q.put(v)

    if len(order) != len(in_deg):
        print('There is a cycle in includes!')
        sys.exit(1)

    print('Order of files:', *order, sep='\n')
    return order


def generate_code(args, order: list[str], codes: dict[str, str]):
    code = []

    code.append("""#undef _GLIBCXX_DEBUG // disable run-time bound checking, etc
#pragma GCC optimize("Ofast,inline")
#define NDEBUG
#pragma GCC option("arch=native","tune=native","no-zeroupper") //Enable AVX
#pragma GCC target("bmi,bmi2,lzcnt,popcnt")                      // bit manipulation
#pragma GCC target("movbe")                                      // byte swap
#pragma GCC target("aes,pclmul,rdrnd")                           // encryption
#pragma GCC target("avx,avx2,f16c,fma,sse3,ssse3,sse4.1,sse4.2") // SIMD

// Caution! Include headers *after* compile options.
#include <x86intrin.h> //AVX/SSE Extensions
#include <immintrin.h>
    """)

    for file in order:
        code.append(f"// {file}\n")
        code.append(codes[file])

    if args.out:
        with open(os.path.join("CG-codes", args.out), 'w+') as file:
            file.writelines(code)

    pyperclip.copy("".join(code))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
        Program to merge many c++ files into one .cpp file.
        It splits main file into 'before first include' and after.
        First it write 'before first include' to out, then all dependencies
        and finally main after first include.
    """
    )

    parser.add_argument("main", type=str,
                        help="Main file to merge.")

    parser.add_argument("--out", type=str,
                        help="Name of file to save the merged result.")

    args = parser.parse_args()

    graph, codes = generate_graph_and_codes(args)
    order = generate_order(graph)
    generate_code(args, order, codes)


if __name__ == "__main__":
    main()
