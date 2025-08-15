#pragma once
#include "mailbox.hpp"
#include <set>
#include <functional>

class GpuRank;
class CommGroup;

class ThreadBlock {
public:
    void Initialize(tinyxml2::XMLElement* tb_elem, std::shared_ptr<GpuRank> my_rank);
    void LoadInstructions(tinyxml2::XMLElement* tb_elem);
    const std::vector<Instruction>& getInstructions() const;
    void ExecuteSingleStep(int step);
    void ExecuteInstructions();

private:
    int tbid, send_peer, recv_peer, chan_id;
    std::shared_ptr<Mailbox> send_mailbox;
    std::shared_ptr<Mailbox> recv_mailbox;
    std::shared_ptr<GpuRank> gpu_rank;
    std::vector<Instruction> instructions;
};

class GpuRank: public std::enable_shared_from_this<GpuRank> {
public:
    struct InstructionStep {
        int tbid;
        int step;
        bool operator<(const InstructionStep& other) const {
            return std::tie(tbid, step) < std::tie(other.tbid, other.step);
        }
    };

    std::shared_ptr<ThreadBlock> getThreadBlock(int tbid) const;
    void InitializeThreadBlocks(tinyxml2::XMLElement* rank_elem, std::shared_ptr<CommGroup> my_group);
    void ExecuteThreadBlocks();
    void InitData(std::function<ChunkDataType(int, size_t)> init_func, size_t input_buff_size);
    void CheckData(std::function<ChunkDataType(int, size_t)> check_func, size_t output_buff_size) const;

private:
    int rank;
    std::shared_ptr<CommGroup> comm_group;
    std::vector<std::shared_ptr<ThreadBlock>> threadblocks;
    
    std::set<InstructionStep> instructionSteps; // To track executed steps that other threadblocks depend on
    mutable std::mutex instructionMutex;

    /**
     * Buffers Should not be protected, though maybe concurrently accessed by multiple threadblocks
     * Any read-write hazard should be avoided by dependency in XML instructions
     */
    std::map<BufferType, std::vector<ChunkDataType>> buffers;

    friend class ThreadBlock;
};

class CommGroup: public std::enable_shared_from_this<CommGroup> {
public:
    size_t getNumRanks() const;
    std::shared_ptr<GpuRank> getRank(int rank_id) const;
    std::shared_ptr<MailboxManager> getMailboxManager() const;
    void InitializeRanks(tinyxml2::XMLElement* root_elem);
    void ExecuteRanks();
    /**
     * @brief Initializes the data in the buffers of each rank.
     * @param init_func A function that takes a rank ID and an input buffer index, and returns the initial data for that chunk.
     */
    void InitData(std::function<ChunkDataType(int, size_t)> init_func, size_t input_buff_size);
    /**
     * @brief Checks the data in the buffers of each rank.
     * @param check_func A function that takes a rank ID and an output buffer index, and checks the data for that chunk.
     */
    void CheckData(std::function<ChunkDataType(int, size_t)> check_func, size_t output_buff_size) const;

private:
    std::vector<std::shared_ptr<GpuRank>> ranks;
    std::shared_ptr<MailboxManager> mailboxManager;

    friend class GpuRank;
    friend class ThreadBlock;
};