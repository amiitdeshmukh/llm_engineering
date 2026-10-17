#include <cstdio>
#include <chrono>

int main() {
    const auto start = std::chrono::steady_clock::now();

    double result = 1.0;
    for (int i = 1; i <= 200000000; ++i) {
        double j = static_cast<double>(i * 4 - 1);
        result -= 1.0 / j;
        j = static_cast<double>(i * 4 + 1);
        result += 1.0 / j;
    }
    result *= 4.0;

    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    std::printf("Result: %.12f\n", result);
    std::printf("Execution Time: %.6f seconds\n", elapsed);
    return 0;
}