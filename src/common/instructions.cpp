#include "instructions.hpp"
#include <string>

static OpType opStrToOp(const char *op_str) {
    if (strcmp(op_str, "cpy") == 0) {
        return OpType::copy;
    } else if (strcmp(op_str, "s") == 0) {
        return OpType::send;
    } else if (strcmp(op_str, "r") == 0) {
        return OpType::recv;
    } else if (strcmp(op_str, "nop") == 0) {
        return OpType::nop;
    } else if (strcmp(op_str, "rcs") == 0) {
        return OpType::rcs;
    } else {
        throw std::runtime_error("Unknown operation " + std::string(op_str));
    }
};

BufferType bufferStrToBuffer(const char *buf_str) {
    if (strcmp(buf_str, "i") == 0) {
        return BufferType::input;
    } else if (strcmp(buf_str, "o") == 0) {
        return BufferType::output;
    } else if (strcmp(buf_str, "s") == 0) {
        return BufferType::scratch;
    } else {
        throw std::runtime_error("Unknown buffer " + std::string(buf_str));
    }
};

Instruction::Instruction(tinyxml2::XMLElement* step_elem) {
    step = std::stoi(SafeGetAttribute(step_elem, "s"));
    op = opStrToOp(SafeGetAttribute(step_elem, "type"));
    src_buff = bufferStrToBuffer(SafeGetAttribute(step_elem, "srcbuf"));
    src_off = std::stoi(SafeGetAttribute(step_elem, "srcoff"));
    dst_buff = bufferStrToBuffer(SafeGetAttribute(step_elem, "dstbuf"));
    dst_off = std::stoi(SafeGetAttribute(step_elem, "dstoff"));
    num_chunks = std::stoi(SafeGetAttribute(step_elem, "cnt"));
    dep_tbid = std::stoi(SafeGetAttribute(step_elem, "depid"));
    dep_step = std::stoi(SafeGetAttribute(step_elem, "deps"));
    has_dep = std::stoi(SafeGetAttribute(step_elem, "hasdep")) != 0;

    if (op == OpType::rcs) {
        if (src_buff != dst_buff || src_off != dst_off) {
            throw std::runtime_error("For RCS operation, src and dst buffers and offsets must match.");
        }
    }

    if (op != OpType::nop) {
        if (num_chunks <= 0 || num_chunks >= 72) {
            throw std::runtime_error("Number of chunks must be between 1 and 71 (inclusive), got " + std::to_string(num_chunks));
        }
    }
}

std::ostream& operator<<(std::ostream& os, const OpType& op) {
    switch (op) {
        case OpType::send: os << "send"; break;
        case OpType::recv: os << "recv"; break;
        case OpType::copy: os << "copy"; break;
        case OpType::nop: os << "nop"; break;
        case OpType::rcs: os << "rcs"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const BufferType& buf) {
    switch (buf) {
        case BufferType::input: os << "input"; break;
        case BufferType::output: os << "output"; break;
        case BufferType::scratch: os << "scratch"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Instruction& inst) {
    os << "Inst { "
       << "step: " << inst.step << ", "
       << "op: " << inst.op << ", "
       << "src_buff: " << inst.src_buff << ", "
       << "src_off: " << inst.src_off << ", "
       << "dst_buff: " << inst.dst_buff << ", "
       << "dst_off: " << inst.dst_off << ", "
       << "num_chunks: " << inst.num_chunks << ", "
       << "dep_tbid: " << inst.dep_tbid << ", "
       << "dep_step: " << inst.dep_step << ", "
       << "has_dep: " << (inst.has_dep ? "true" : "false") << " "
       << "}";
    return os;
}