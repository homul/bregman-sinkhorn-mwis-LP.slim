#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>   //for time measurements
#include <nlohmann/json.hpp>
#include "solver.hpp"


  template <typename RandomIt>
  void print_n_elements_as_json(RandomIt begin, size_t n, const std::string& name
                              , size_t reserve_size) {
      using T = typename std::iterator_traits<RandomIt>::value_type;
    if (n > 0) {
      std::vector<int> ones_positions; ones_positions.reserve(reserve_size);

      int index = 0;
      std::for_each(begin, begin+n, [&](int val) {
          if (val == 1) ones_positions.push_back(index);
          ++index;
      });

      std::cout << "\""<<name << "\":[";
      std::copy(ones_positions.begin(), ones_positions.end()-1,std::ostream_iterator<int>(std::cout, ","));
      std::cout << *(ones_positions.end()-1) << "]";
    }
  }

  template <typename RandomIt>
  void print_n_elements_as_json_binary(RandomIt begin, size_t n, const std::string& name) {
      using T = typename std::iterator_traits<RandomIt>::value_type;
 
     if (n > 0) {
       std::cout << "\""<<name << "\":[";
            std::copy(begin, begin+n-1,std::ostream_iterator<int>(std::cout, ","));
            std::cout << *(begin+n-1) << "]";
      }
  }

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.json num_greedy_solutions" << std::endl;
        return 1;
    }

    // Input JSON file
    std::string filename = argv[1];
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    // Parse greedy solutions parameter
    int num_greedy_solutions = 0;
    try {
        num_greedy_solutions = std::stoi(argv[2]);
        if (num_greedy_solutions <= 0) {
            std::cerr << "Error: num_greedy_solutions must be a positive integer" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: invalid number of greedy solutions: " << argv[2] << std::endl;
        return 1;
    }

    // Parse JSON
    nlohmann::json j;
    try {
        infile >> j;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return 1;
    }

    // Extract nodes
    std::vector<double> nodes;
    if (j.contains("nodes") && j["nodes"].is_array()) {
        for (const auto& node_cost : j["nodes"]) {
            if (!node_cost.is_number()) {
                std::cerr << "Invalid node cost format" << std::endl;
                return 1;
            }
            nodes.push_back(node_cost.get<double>());
        }
    } else {
        std::cerr << "Error: JSON missing 'nodes' array" << std::endl;
        return 1;
    }

    // Extract cliques
    std::vector<std::vector<mwis::index>> cliques;
    if (j.contains("cliques") && j["cliques"].is_array()) {
        for (const auto& clique : j["cliques"]) {
            if (!clique.is_array()) {
                std::cerr << "Invalid clique format" << std::endl;
                return 1;
            }
            std::vector<mwis::index> indices;
            for (const auto& idx : clique) {
                if (!idx.is_number_unsigned()) {
                    std::cerr << "Invalid node index in clique" << std::endl;
                    return 1;
                }
                indices.push_back(idx.get<mwis::index>());
            }
            cliques.push_back(std::move(indices));
        }
    } else {
        std::cerr << "Error: JSON missing 'cliques' array" << std::endl;
        return 1;
    }

std::cout <<std::endl<< "======Provide the problem to the solver =====================" <<std::endl;
    // Feed into solver   
    mwis::solver solver;
    for (double cost : nodes) {
        solver.add_node(cost);
    }
    for (const auto& clique : cliques) {
        solver.add_clique(clique);
    }

    solver.finalize();
std::cout <<std::endl<< "======Problem dimensions =====================" <<std::endl;

    std::cout << "Number of nodes in the original problem = "<<solver.no_orig()<<std::endl;
    std::cout << "Number of cliques = "<<solver.no_cliques()<<std::endl;
    std::cout << "Number of nodes including slacks = "<<solver.no_nodes()<<std::endl;

std::cout <<std::endl<< "======Run the dual solver ====================================" <<std::endl;

    solver.temperature(0.01); // set initial temperature

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    solver.dualRun(  50,          // batch size
                1000000,      // max number of batches
                0.1           // relative duality gap
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Total run-time of the dual solver " << elapsed.count() << " seconds." << std::endl;

std::cout <<std::endl<< "======Generate greedy solutions ================================" <<std::endl;

    std::cout << "Generated " << num_greedy_solutions << " greedy solutions:" <<std::endl;
    std::vector<mwis::index> greedy_solution(solver.no_orig());
    std::cout << "{";
    for (int i = 0; i < num_greedy_solutions; ++i)
    {
        solver.generate_greedy_solution(greedy_solution.begin(),greedy_solution.end());

        
        // uncomment the line below and comment out the next one if you need binary representation
        // print_n_elements_as_json_binary(greedy_solution.begin(), greedy_solution.size(), "solution-" + std::to_string(i + 1));
        print_n_elements_as_json(greedy_solution.begin(), greedy_solution.size(), "solution-" + std::to_string(i + 1),solver.no_cliques());
        if (i < num_greedy_solutions-1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;

#ifndef NDEBUG //code below runs only in debug mode, as it produces a lot of output

std::cout <<std::endl<< "======Output the best found relaxed solution (the best amoung projected and integer solutions) =====" <<std::endl;
    std::vector<mwis::cost_exp> relaxed_solution(solver.no_nodes()); // number of nodes including slack variables
    solver.assignment_relaxed(relaxed_solution.begin(),relaxed_solution.end());

    std::cout << "Best found relaxed solution:" <<std::endl;
    std::copy(relaxed_solution.begin(), relaxed_solution.end(),std::ostream_iterator<mwis::cost_exp>(std::cout, ","));
    std::cout <<std::endl;

    std::cout << "Objective value of the best relaxed solution =" << solver.primal_relaxed() << std::endl;

std::cout <<std::endl<< "======Output the latest relaxed solution found by truncation projection ====================" <<std::endl;
    std::vector<mwis::cost_exp> relaxed_solution_projected(solver.no_nodes()); // number of nodes including slack variables
    solver.assignment_relaxed_projected(relaxed_solution_projected.begin(),relaxed_solution_projected.end());

    std::cout << "Latest found relaxed solution by projection:" <<std::endl;
    std::copy(relaxed_solution_projected.begin(), relaxed_solution_projected.end(),std::ostream_iterator<mwis::cost_exp>(std::cout, ","));
    std::cout <<std::endl;

    std::cout << "Objective value of the latest relaxed solution by projection =" << solver.primal_relaxed_projected() << std::endl;

std::cout <<std::endl<< "======Output the reduced costs ====================================================" <<std::endl;
    std::vector<mwis::cost> reduced_costs(solver.no_nodes());
    solver.reduced_costs(reduced_costs.begin(),reduced_costs.end());
    
    std::cout << "Reduced costs:" <<std::endl;
    std::copy(reduced_costs.begin(), reduced_costs.end(),std::ostream_iterator<mwis::cost>(std::cout, ","));
    std::cout <<std::endl;

    std::cout << "The sum of dual variables = "<<solver.constant() <<std::endl;
#endif

    return 0;
}
