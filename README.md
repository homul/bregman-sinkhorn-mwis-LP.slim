# Slim Version of the MWIS Code

This repository contains a **slim version** of the MWIS (Maximum Weight Independent Set) code from the paper:

> **S. Haller, B. Savchynskyy**
> *A Bregman-Sinkhorn Algorithm for the Maximum Weight Independent Set Problem*
> [arXiv:2408.02086](https://arxiv.org/pdf/2408.02086)

The **full code** used in the publication, along with other related data, is available at:
👉 [https://vislearn.github.io/libmpopt/mwis2024/](https://vislearn.github.io/libmpopt/mwis2024/)

---

## About This Version

This is the **LP (non-ILP)** version used for **METAMIS comparison**, extracted from:


```
https://www.stha.de/shares/mwis2024/mwis2024_code.tar.zst/
mwis2024_code/integer/variant4-dualgap-heuristicspeedup-mindual=0.1-integer/
```

### Features

This code includes:

* Duality-gap-based smoothing scheduling
* Heuristic truncation for sparsification

It can be used as a **specialized LP solver** that returns:

* An approximate **primal LP solution**
* **Reduced costs**

Additionally, it may produce:

* **Diverse integer solutions** using a randomized greedy algorithm

### Limitations

This slim version **does not include**:

* Calls to the **optimal recombination / crossover** routines that merge two integer solutions

Therefore, it is **not suitable for solving the full MWIS problem directly**.
See `main.cpp` for a usage example.

---

## 🛠️ Installation and Compilation Guide

### 1. Compile the MWIS Solver

#### 1.1. Download Dependencies

Download the [nlohmann/json](https://github.com/nlohmann/json/releases) library and place it into the `/external` folder. The latter must be created separately.
Check the corresponding include path to `<nlohmann/json.hpp>` in your `CMakeLists.txt`.

#### 1.2. Build the Project

```bash
mkdir build && cd build
cmake ..
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Use `cmake -DCMAKE_BUILD_TYPE=Debug ..` for the Debug mode.

---

### 2. Run the Solver

Use the following command:

```bash
./main <input-file.json> <number-of-greedy-solutions> > <output-file-name>
```

#### Examples

```bash
./main ../tests/test.json 10
./main ../tests/test3.json 16
```
The respective outputs in Release mode can be found in  `../tests/output.txt` and `../tests/output3.txt`.

---

### 2.1. Notes

1. The **second parameter** defines how many greedy solutions will be generated.
   The program stops after producing the specified number of solutions.
2. The **precision** of solving the relaxed problem is controlled in `main.cpp` in the line containing the comment `"relative duality gap"`:

   ```cpp
   solver.run(
       50,          // batch size
       1000000,     // max number of batches
       0.1          // relative duality gap
   );
   ```

   You can adjust the value `0.1` to balance **runtime** and **solution quality**.

---
