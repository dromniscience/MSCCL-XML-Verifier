#include "threadblock.hpp"

void ThreadBlock::Initialize(tinyxml2::XMLElement* tb_elem, std::shared_ptr<GpuRank> my_rank) {
    tbid = std::stoi(SafeGetAttribute(tb_elem, "id"));
    send_peer = std::stoi(SafeGetAttribute(tb_elem, "send"));
    recv_peer = std::stoi(SafeGetAttribute(tb_elem, "recv"));
    chan_id = std::stoi(SafeGetAttribute(tb_elem, "chan"));
    
    gpu_rank = my_rank;
    if (send_peer >= 0) {
        if (send_peer == gpu_rank->rank) {
            throw std::runtime_error("ThreadBlock " + std::to_string(tbid) + " in rank " + std::to_string(gpu_rank->rank) + " cannot send to itself.");
        }
        gpu_rank->comm_group->mailboxManager->getSendMailbox(gpu_rank->rank, send_peer, chan_id, send_mailbox);
    }
    if (recv_peer >= 0) {
        if (recv_peer == gpu_rank->rank) {
            throw std::runtime_error("ThreadBlock " + std::to_string(tbid) + " in rank " + std::to_string(gpu_rank->rank) + " cannot receive from itself.");
        }
        gpu_rank->comm_group->mailboxManager->getRecvMailbox(recv_peer, gpu_rank->rank, chan_id, recv_mailbox);
    }

    LoadInstructions(tb_elem);
}

void ThreadBlock::LoadInstructions(tinyxml2::XMLElement* tb_elem) {
    for (tinyxml2::XMLElement* step_elem = tb_elem->FirstChildElement("step");
         step_elem != nullptr;
         step_elem = step_elem->NextSiblingElement("step")) {
        instructions.push_back(Instruction(step_elem));
        if (instructions.back().step != static_cast<int>(instructions.size()) - 1) {
            throw std::runtime_error("Instructions in ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + " are not in the correct order.");
        }
        if (instructions.back().step >= 256) {
            throw std::runtime_error("Number of instructions exceeds the limit of 256 in ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
        }
    }
    // Check that no recv proceeds rcs instructions and no send follows rcs instructions
    size_t first_recv = instructions.size();
    size_t last_send = 0;
    size_t first_rcs = instructions.size();
    size_t last_rcs = 0;
    for (size_t i = 0; i < instructions.size(); ++i) {
        if (instructions[i].op == OpType::recv) {
            first_recv = std::min(first_recv, i);
        }
        if (instructions[i].op == OpType::send) {
            last_send = std::max(last_send, i);
        }
        if (instructions[i].op == OpType::rcs) {
            first_rcs = std::min(first_rcs, i);
            last_rcs = std::max(last_rcs, i);
        }
    }
    if (first_recv < last_rcs) {
        throw std::runtime_error("A recv instruction cannot precede an rcs instruction in ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
    }
    if (last_send > first_rcs) {
        throw std::runtime_error("A send instruction cannot be after an rcs instruction in ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
    }
}

const std::vector<Instruction>& ThreadBlock::getInstructions() const {
    return instructions;
}

void ThreadBlock::ExecuteSingleStep(int step) {
    const Instruction &inst = instructions.at(step);
    // Check if the dependency is met
    if (inst.dep_tbid >= 0 || inst.dep_step >= 0) {
        if (inst.dep_tbid < 0 || inst.dep_step < 0) {
            throw std::runtime_error("Invalid dependency in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
        }
        bool timeout = true;
        for (int tries = 0; tries < MAX_TRIES; ++tries) {
            {
                std::lock_guard<std::mutex> lock(gpu_rank->instructionMutex);
                if (gpu_rank->instructionSteps.count({inst.dep_tbid, inst.dep_step}) > 0) {
                    timeout = false;
                    break;
                }
            }
            std::this_thread::sleep_for(SLEEP_TIME);
        }
        if (timeout) {
            throw std::runtime_error("Dependency not met in time for instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
        }
    }
    // Execute the instruction based on its operation type
    switch (inst.op) {
        case OpType::copy: {
            const auto &src_buffer = gpu_rank->buffers[inst.src_buff];
            auto &dst_buffer = gpu_rank->buffers[inst.dst_buff];
            if (inst.src_off < 0 || inst.src_off + inst.num_chunks > src_buffer.size() ||
                inst.dst_off < 0 || inst.dst_off + inst.num_chunks > dst_buffer.size()) {
                throw std::runtime_error("Invalid buffer offsets in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            // std::lock_guard<std::mutex> lock(gpu_rank->bufferMutex);
            std::copy(src_buffer.begin() + inst.src_off, src_buffer.begin() + inst.src_off + inst.num_chunks,
                      dst_buffer.begin() + inst.dst_off);
            break;
        }
        case OpType::recv: {
            Message msg;
            if (!recv_mailbox->receiveMessage(msg)) {
                throw std::runtime_error("Failed to receive message in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            auto &dst_buffer = gpu_rank->buffers[inst.dst_buff];
            if (inst.dst_off < 0 || inst.dst_off + msg.chunks.size() > dst_buffer.size()) {
                throw std::runtime_error("Invalid destination buffer offset in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            if (msg.src_buff != inst.src_buff || msg.src_off != inst.src_off || msg.chunks.size() != inst.num_chunks ||
                msg.dst_buff != inst.dst_buff || msg.dst_off != inst.dst_off) {
                throw std::runtime_error("Message mismatch in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            // std::lock_guard<std::mutex> lock(gpu_rank->bufferMutex);
            std::copy(msg.chunks.begin(), msg.chunks.end(), dst_buffer.begin() + inst.dst_off);
            break;
        }
        case OpType::send: {
            Message msg;
            msg.chunks.resize(inst.num_chunks);
            msg.src_buff = inst.src_buff;
            msg.src_off = inst.src_off;
            msg.dst_buff = inst.dst_buff;
            msg.dst_off = inst.dst_off;

            const auto &src_buffer = gpu_rank->buffers[inst.src_buff];
            if (inst.src_off < 0 || inst.src_off + inst.num_chunks > src_buffer.size()) {
                throw std::runtime_error("Invalid source buffer offset in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            {
                // std::lock_guard<std::mutex> lock(gpu_rank->bufferMutex);
                std::copy(src_buffer.begin() + inst.src_off, src_buffer.begin() + inst.src_off + inst.num_chunks, msg.chunks.begin());
            }
            send_mailbox->sendMessage(msg);
            break;
        }
        case OpType::rcs: {
            Message msg;
            if (!recv_mailbox->receiveMessage(msg)) {
                throw std::runtime_error("Failed to receive message in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            auto &dst_buffer = gpu_rank->buffers[inst.dst_buff];
            if (inst.dst_off < 0 || inst.dst_off + msg.chunks.size() > dst_buffer.size()) {
                throw std::runtime_error("Invalid destination buffer offset in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            if (msg.src_buff != inst.src_buff || msg.src_off != inst.src_off || msg.chunks.size() != inst.num_chunks ||
                msg.dst_buff != inst.dst_buff || msg.dst_off != inst.dst_off) {
                throw std::runtime_error("Message mismatch in instruction step " + std::to_string(step) + " of ThreadBlock " + std::to_string(tbid) + " Rank " + std::to_string(gpu_rank->rank) + ".");
            }
            {
                // std::lock_guard<std::mutex> lock(gpu_rank->bufferMutex);
                std::copy(msg.chunks.begin(), msg.chunks.end(), dst_buffer.begin() + inst.dst_off);
            }
            msg.src_buff = msg.dst_buff;
            msg.src_off = msg.dst_off;
            {
                // std::lock_guard<std::mutex> lock(gpu_rank->bufferMutex);
                std::copy(dst_buffer.begin() + inst.dst_off, dst_buffer.begin() + inst.dst_off + msg.chunks.size(), msg.chunks.begin());
            }
            send_mailbox->sendMessage(msg);
            break;
        }
        case OpType::nop:
            break;
    }

    // Update instruction step if other instructions depend on it
    if (inst.has_dep) {
        std::lock_guard<std::mutex> lock(gpu_rank->instructionMutex);
        gpu_rank->instructionSteps.insert({tbid, step});
    }
}

void ThreadBlock::ExecuteInstructions() {
    int num_steps = instructions.size();
    SleepForRandomTime(SLEEP_TIME.count() * MAX_TRIES / 1000.0);
    for (int step = 0; step < num_steps; ++step) {
        ExecuteSingleStep(step);
    }
}

void ThreadBlock::SleepForRandomTime(double max_us) {
    if (max_us <= 0) return;
    std::uniform_real_distribution<double> dist(0.0, max_us);
    double sleep_time = dist(this->rng);
    std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(sleep_time));
}

std::shared_ptr<ThreadBlock> GpuRank::getThreadBlock(int tbid) const {
    return threadblocks.at(tbid);
}

void GpuRank::InitializeThreadBlocks(tinyxml2::XMLElement* rank_elem, std::shared_ptr<CommGroup> my_group) {
    rank = std::stoi(SafeGetAttribute(rank_elem, "id"));
    comm_group = my_group;

    int i_chunks = std::stoi(SafeGetAttribute(rank_elem, "i_chunks"));
    int o_chunks = std::stoi(SafeGetAttribute(rank_elem, "o_chunks"));
    int s_chunks = std::stoi(SafeGetAttribute(rank_elem, "s_chunks"));

    buffers[BufferType::input].resize(i_chunks);
    buffers[BufferType::output].resize(o_chunks);
    buffers[BufferType::scratch].resize(s_chunks);

    int num_tbs = rank_elem->ChildElementCount("tb");
    if (num_tbs >= 78) {
        throw std::runtime_error("Number of threadblocks exceeds the limit of 78 in rank " + std::to_string(rank) + ".");
    }
    std::vector<tinyxml2::XMLElement*> tb_elem(num_tbs);
    for (int i = 0; i < num_tbs; ++i) {
        if (i == 0) {
            tb_elem[i] = rank_elem->FirstChildElement("tb");
        } else {
            tb_elem[i] = tb_elem[i - 1]->NextSiblingElement("tb");
        }
        if (!tb_elem[i]) {
            throw std::runtime_error("Not enough threadblocks in rank " + std::to_string(rank) + ".");
        }
        int tbid = std::stoi(SafeGetAttribute(tb_elem[i], "id"));
        if (tbid != i) {
            throw std::runtime_error("Threadblocks in rank " + std::to_string(rank) + " are not in the correct order.");
        }
        threadblocks.push_back(std::make_shared<ThreadBlock>());
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < num_tbs; ++i) {
        threads.emplace_back([this, i, tb_elem]() {
            this->threadblocks[i]->Initialize(tb_elem[i], shared_from_this());
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    // Check XML node number under each GPU
    size_t xml_node_nums = 1 + threadblocks.size();
    for (const auto& tb : threadblocks) {
        xml_node_nums += tb->getInstructions().size();
    }
    if (xml_node_nums > 4096) {
        throw std::runtime_error("Number of XML nodes (" + std::to_string(xml_node_nums) + ") exceeds the limit of 4096 in rank " + std::to_string(rank) + ".");
    }
}

void GpuRank::ExecuteThreadBlocks() {
    int num_tbs = threadblocks.size();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_tbs; ++i) {
        threads.emplace_back([this, i]() {
            this->threadblocks[i]->ExecuteInstructions();
        });
    }
    for (auto& th : threads) {
        th.join();
    }
}

void GpuRank::InitData(std::function<ChunkDataType(int, size_t)> init_func, size_t input_buff_size) {
    if (buffers[BufferType::input].size() != input_buff_size) {
        throw std::runtime_error("Input buffer size mismatch in rank " + std::to_string(rank) + ".");
    }
    for (size_t i = 0; i < input_buff_size; ++i) {
        buffers[BufferType::input][i] = init_func(rank, i);
    }
}

void GpuRank::CheckData(std::function<ChunkDataType(int, size_t)> check_func, size_t output_buff_size) const {
    if (buffers.at(BufferType::output).size() != output_buff_size) {
        throw std::runtime_error("Output buffer size mismatch in rank " + std::to_string(rank) + ".");
    }
    for (size_t i = 0; i < output_buff_size; ++i) {
        ChunkDataType expected = check_func(rank, i);
        if (buffers.at(BufferType::output)[i] != expected) {
            throw std::runtime_error("Data mismatch in output buffer at index " + std::to_string(i) + " in rank " + std::to_string(rank) + ": Expected " + expected + ", but got " + buffers.at(BufferType::output)[i] + ".");
        }
    }
}

size_t CommGroup::getNumRanks() const {
    return ranks.size();
}

std::shared_ptr<GpuRank> CommGroup::getRank(int rank_id) const {
    return ranks.at(rank_id);
}

std::shared_ptr<MailboxManager> CommGroup::getMailboxManager() const {
    return mailboxManager;
}

void CommGroup::InitializeRanks(tinyxml2::XMLElement* root_elem) {
    int num_ranks = std::stoi(SafeGetAttribute(root_elem, "ngpus"));
    int num_chans = std::stoi(SafeGetAttribute(root_elem, "nchannels"));
    if (num_chans > 32) {
        throw std::runtime_error("Number of channels exceeds the limit of 32.");
    }
    int num_chunks = std::stoi(SafeGetAttribute(root_elem, "nchunksperloop"));
    if (num_chunks & (num_chunks - 1)) {
        throw std::runtime_error("Number of chunks should be a power of 2, got " + std::to_string(num_chunks) + ".");
    }
    int outofplace = std::stoi(SafeGetAttribute(root_elem, "outofplace"));
    if (outofplace == 0) {
        throw std::runtime_error("Only out-of-place collective is supported.");
    }

    mailboxManager = std::make_shared<MailboxManager>();

    std::vector<tinyxml2::XMLElement*> rank_elem(num_ranks);
    for (int i = 0; i < num_ranks; ++i) {
        if (i == 0) {
            rank_elem[i] = root_elem->FirstChildElement("gpu");
        } else {
            rank_elem[i] = rank_elem[i - 1]->NextSiblingElement("gpu");
        }
        if (!rank_elem[i]) {
            throw std::runtime_error("Not enough ranks in XML.");
        }
        int rank_id = std::stoi(SafeGetAttribute(rank_elem[i], "id"));
        if (rank_id != i) {
            throw std::runtime_error("Ranks are not in the correct order in XML.");
        }
        ranks.push_back(std::make_shared<GpuRank>());
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < num_ranks; ++i) {
        threads.emplace_back([this, i, rank_elem]() {
            this->ranks[i]->InitializeThreadBlocks(rank_elem[i], shared_from_this());
        });
    }
    for (auto& th : threads) {
        th.join();
    }
}

void CommGroup::ExecuteRanks() {
    int num_ranks = ranks.size();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_ranks; ++i) {
        threads.emplace_back([this, i]() {
            this->ranks[i]->ExecuteThreadBlocks();
        });
    }
    for (auto& th : threads) {
        th.join();
    }
}

void CommGroup::InitData(std::function<ChunkDataType(int, size_t)> init_func, size_t input_buff_size) {
    for (auto &rank: ranks) {
        rank->InitData(init_func, input_buff_size);
    }
}

void CommGroup::CheckData(std::function<ChunkDataType(int, size_t)> check_func, size_t output_buff_size) const {
    for (const auto &rank : ranks) {
        rank->CheckData(check_func, output_buff_size);
    }
}