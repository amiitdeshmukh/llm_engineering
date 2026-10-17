You **do not need to install a C++ compiler**. Your system already has Apple Clang installed through the Xcode Command Line Tools. The `g++` listed in your report is Apple Clang, not GNU GCC.

From the directory containing `main.cpp`, use:

```python
compile_command = [
    "/usr/bin/clang++",
    "-std=c++20",
    "-O3",
    "main.cpp",
    "-o",
    "main",
]
compile_result = subprocess.run(
    compile_command, check=True, text=True, capture_output=True
)

run_command = ["./main"]
run_result = subprocess.run(
    run_command, check=True, text=True, capture_output=True
)
return run_result.stdout
```

`-O3` enables compiler optimizations for runtime performance. The compiled program will be named `main` in the current directory. If your Python process is running from a different directory, set `cwd` in `subprocess.run` or use full paths to the source and executable.