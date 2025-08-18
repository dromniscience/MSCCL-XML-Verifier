#include "common/threadblock.hpp"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_xml_file> <run_iters>" << std::endl;
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

    if (SafeGetAttribute(root_elem, "coll") != std::string("allgather")) {
        std::cerr << "Error: Only allgather collective is supported." << std::endl;
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

    auto init_func = [chunk_factor](int rank_id, size_t index) -> ChunkDataType {
        return std::to_string(rank_id) + "_" + std::to_string(index % chunk_factor);
    };
    auto check_func = [chunk_factor](int rank_id, size_t index) -> ChunkDataType {
        return std::to_string(index / chunk_factor) + "_" + std::to_string(index % chunk_factor);
    };

    int run_iters = std::stoi(argv[2]);
    for (int i = 0; i < run_iters; i++) {
        if (i % 10 == 0) {
            std::cout << "Running iteration " << i << "/" << run_iters << std::endl;
        }
        comm_group->InitData(init_func, chunk_factor);
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