#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "defs.h"
#include "gen_compare_gpu.h"

Base randomBase() {
    static const Base bases[] = {'C', 'T', 'G', 'A'};
    return bases[rand() % 4];
}

GeneSequence randomGeneSequence(int size) {
    GeneSequence genes;
    genes.reserve(size);
    for (int n = 0; n < size; ++n) {
        genes.push_back(randomBase());
    }
    return genes;
}

void showResults(GeneSequence& genes1, GeneSequence& genes2, std::vector<int>& matches, int last) {
    int total = matches.size() / 3;
    int start = std::max(0, total - last) * 3;
    for (size_t c = start; c < matches.size(); c += 3) {
        std::cout << "g1: " << matches[c]
                  << " g2: " << matches[c + 1]
                  << " len: " << matches[c + 2] << " ";
        for (int i = 0; i < matches[c + 2]; ++i)
            std::cout << static_cast<char>(genes1[matches[c] + i]);
        std::cout << " ";
        for (int i = 0; i < matches[c + 2]; ++i)
            std::cout << static_cast<char>(genes2[matches[c + 1] + i]);
        std::cout << '\n';
    }
}

std::vector<int> findMatches(GeneSequence& genes1, GeneSequence& genes2, int min) {
    std::vector<int> result;

    int maxcg1 = genes1.size() - min;
    int maxcg2 = genes2.size() - min;

    for (int cg1 = 0; cg1 < maxcg1; ++cg1) {
        for (int cg2 = 0; cg2 < maxcg2; ++cg2) {
            int icg1 = cg1;
            int icg2 = cg2;
            while (icg1 < static_cast<int>(genes1.size()) &&
                   icg2 < static_cast<int>(genes2.size()) &&
                   genes1[icg1] == genes2[icg2]) {
                ++icg1;
                ++icg2;
            }

            int gl = icg1 - cg1;
            if (gl >= min) {
                result.push_back(cg1);
                result.push_back(cg2);
                result.push_back(gl);
            }
        }
    }

    return result;
}

int main() {
    // srand(time(0));  // Uncomment for different sequences per run

    GeneSequence genes1 = randomGeneSequence(10000);
    GeneSequence genes2 = randomGeneSequence(10000);

    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> matches = findMatches(genes1, genes2, 9);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpuMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // GPU
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<int> gpuMatches = findMatchesGPU(genes1, genes2, 9);
    auto t3 = std::chrono::high_resolution_clock::now();
    float gpuMs = std::chrono::duration<float, std::milli>(t3 - t2).count();

    // Last 3 matches from each
    std::cout << "CPU last 3 matches:\n";
    showResults(genes1, genes2, matches, 3);
    std::cout << "GPU last 3 matches:\n";
    showResults(genes1, genes2, gpuMatches, 3);

    std::cout << "\nCPU: " << matches.size() / 3 << " matches in "
              << cpuMs / 1000.0f << " s\n";
    std::cout << "GPU: " << gpuMatches.size() / 3 << " matches in "
              << gpuMs / 1000.0f << " s\n";
    std::cout << "Speedup: " << cpuMs / gpuMs << "x\n";

    return 0;
}
