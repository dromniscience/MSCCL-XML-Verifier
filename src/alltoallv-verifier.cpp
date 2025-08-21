#include "common/threadblock.hpp"
#include <fstream>
#include <sstream>
#include <cassert>

/**
 * @brief Computes the accumulated row sums of the traffic matrix in the form num_ranks * num_ranks
 * 
 * For example,
 * 0  1  2    0  1  3
 * 3  4  5 => 3  7  12
 * 6  7  8    6  13 21
 */
void ComputeAccumulateRowSums(const size_t *traffic_matrix, size_t *acc_row_sums, const int num_ranks) {
    for (int i = 0; i < num_ranks * num_ranks; i += num_ranks) {
        acc_row_sums[i] = traffic_matrix[i];
        for (int j = 1; j < num_ranks; ++j) {
            acc_row_sums[i + j] = acc_row_sums[i + j - 1] + traffic_matrix[i + j];
        }
    }
}

/**
 * @brief Computes the accumulated column sums of the traffic matrix in the form num_ranks * num_ranks
 * 
 * For example,
 * 0  1  2    0  1  2
 * 3  4  5 => 3  5  7
 * 6  7  8    9  12 15
 */
void ComputeAccumulateColSums(const size_t *traffic_matrix, size_t *acc_col_sums, const int num_ranks) {
    std::copy(traffic_matrix, traffic_matrix + num_ranks, acc_col_sums);
    for (int i = num_ranks; i < num_ranks * num_ranks; i += num_ranks) {
        for (int j = 0; j < num_ranks; ++j) {
            acc_col_sums[i + j] = acc_col_sums[i - num_ranks + j] + traffic_matrix[i + j];
        }
    }
}

/**
 * @brief Reads all-to-all traffic from a CSV file and populates the traffic matrix.
 * 
 * Each entry (i,j) in the traffic matrix should be the number of chunks (rather than the amount
 * of data) sent from rank i to rank j.
 */
void ReadAllToAllTraffic(std::ifstream &traffic_file, int num_ranks, size_t chunk_factor, size_t *traffic_matrix) {
    for (int i = 0; i < num_ranks; ++i) {
        std::string line;
        if (!std::getline(traffic_file, line)) {
            throw std::runtime_error("Error reading traffic file: insufficient data for rank " + std::to_string(i));
        }
        std::istringstream ss(line);
        std::string cell;
        std::vector<std::string> row_data; // To store cells of the current row

        while (std::getline(ss, cell, ',')) {
            row_data.push_back(cell);
        }
        if (row_data.size() != num_ranks) {
            throw std::runtime_error("Error reading traffic file: expected " + std::to_string(num_ranks) + " columns, got " + std::to_string(row_data.size()));
        }
        for (int j = 0; j < num_ranks; ++j) {
            traffic_matrix[i * num_ranks + j] = std::stoul(row_data[j]);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_xml_file> <run_iters> <traffic_csv_file>" << std::endl;
        return 1;
    }
    tinyxml2::XMLDocument doc;
    doc.LoadFile(argv[1]);
    if (doc.Error()) {
        std::cerr << "Error loading XML file: " << doc.ErrorIDToName(doc.ErrorID()) << std::endl;
        return 1;
    }
    tinyxml2::XMLElement* root_elem = doc.RootElement();
    std::shared_ptr<CommGroup> comm_group = std::make_shared<CommGroup>();
    comm_group->InitializeRanks(root_elem);

    // Update: Not typo, required by CCF test
    if (SafeGetAttribute(root_elem, "coll") != std::string("allreduce")) {
        std::cerr << "Error: Only alltoall collective is supported (coll should be \"allreduce\" in the xml)." << std::endl;
        return 1;
    }

    const int num_ranks = static_cast<int>(comm_group->getNumRanks());
    const int chunk_factor = static_cast<int>(comm_group->getChunkFactor());
    const int num_chunks = static_cast<int>(comm_group->getNumChunks());
    std::cout << "Initialized " << num_ranks << " ranks, " << num_chunks << " chunks, chunk factor " << chunk_factor << std::endl;

    if (!comm_group->getMailboxManager()->checkNoPendingConnections()) {
        std::cerr << "Error: There are pending connections in the mailbox manager." << std::endl;
        return 1;
    }
    if (!comm_group->getMailboxManager()->checkChannelLayout()) {
        std::cerr << "Error: Invalid channel layout in the mailbox manager." << std::endl;
        return 1;
    }
    std::cout << "Channels built." << std::endl;

    // Prepare traffic matrix
    std::ifstream traffic_file(argv[3]);
    if (!traffic_file.is_open()) {
        std::cerr << "Error opening traffic file: " << argv[3] << std::endl;
        return 1;
    }
    std::vector<size_t> traffic_matrix(num_ranks * num_ranks);
    std::vector<size_t> acc_row_sums(num_ranks * num_ranks);
    std::vector<size_t> acc_col_sums(num_ranks * num_ranks);
    ReadAllToAllTraffic(traffic_file, num_ranks, chunk_factor, traffic_matrix.data());
    traffic_file.close();

    ComputeAccumulateRowSums(traffic_matrix.data(), acc_row_sums.data(), num_ranks);
    ComputeAccumulateColSums(traffic_matrix.data(), acc_col_sums.data(), num_ranks);

    for (int i = num_ranks - 1; i < num_ranks * num_ranks; i += num_ranks) {
        if (acc_row_sums[i] != num_ranks * chunk_factor) {
            std::cerr << "Error: Rank " << i / num_ranks << " has incorrect row sum: " << acc_row_sums[i] << ", expected " << num_ranks * chunk_factor << std::endl;
            return 1;
        }
    }
    for (int i = (num_ranks - 1) * num_ranks; i < num_ranks * num_ranks; ++i) {
        if (acc_col_sums[i] != num_ranks * chunk_factor) {
            std::cerr << "Error: Rank " << i % num_ranks << " has incorrect column sum: " << acc_col_sums[i] << ", expected " << num_ranks * chunk_factor << std::endl;
            return 1;
        }
    }

    std::vector<ChunkDataType> result_data(num_ranks * num_ranks * chunk_factor);
    for (int i = 0; i < num_ranks; ++i) {
        for (int j = 0; j < num_ranks; ++j) {
            // Chunks sent from rank i to rank j
            size_t start_chunk = (j == 0) ? 0 : acc_row_sums[i * num_ranks + (j - 1)];
            size_t end_chunk = acc_row_sums[i * num_ranks + j];
            size_t result_chunk = (i == 0) ? 0 : acc_col_sums[(i - 1) * num_ranks + j];
            for (size_t k = start_chunk; k < end_chunk; ++k, ++result_chunk) {
                result_data[j * num_ranks * chunk_factor + result_chunk] = std::to_string(i) + "_" + std::to_string(k);
            }
        }
    }

    auto init_func = [chunk_factor](int rank_id, size_t index) -> ChunkDataType {
        return std::to_string(rank_id) + "_" + std::to_string(index);
    };
    auto check_func = [result_data, num_ranks, chunk_factor](int rank_id, size_t index) -> ChunkDataType {
        return result_data[rank_id * num_ranks * chunk_factor + index];
    };

    // Run iterations
    int run_iters = std::stoi(argv[2]);
    for (int i = 0; i < run_iters; i++) {
        if (i % 10 == 0) {
            std::cout << "Running iteration " << i << "/" << run_iters << std::endl;
        }
        comm_group->InitData(init_func, num_chunks);
        comm_group->ExecuteRanks();
        comm_group->CheckData(check_func, num_chunks);
        if (!comm_group->getMailboxManager()->checkNoPendingMessage()) {
            std::cerr << "Error: There are pending messages in the mailbox after iteration " << i << "." << std::endl;
            return 1;
        }
    }
    std::cout << "All tests passed." << std::endl;
    return 0;
}
