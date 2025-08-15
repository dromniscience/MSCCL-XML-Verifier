#pragma once
#include <cstddef>
#include <iostream>
#include "tinyxml2.h"

enum class OpType {
    send,
    recv,
    copy,
    nop,
    rcs
};

enum class BufferType {
    input,
    output,
    scratch
};

struct Instruction {
    int step;
    OpType op;
    BufferType src_buff;
    std::ptrdiff_t src_off;
    BufferType dst_buff;
    std::ptrdiff_t dst_off;
    std::size_t num_chunks;
    int dep_tbid;
    int dep_step;
    bool has_dep;

    Instruction(tinyxml2::XMLElement* step_elem);
};

inline const char *SafeGetAttribute(tinyxml2::XMLElement* elem, const char* attr_name) {
    const char* value = elem->Attribute(attr_name);
    if (!value) {
        throw std::runtime_error(std::string("Missing attribute: ") + attr_name);
    }
    return value;
}

std::ostream& operator<<(std::ostream& os, const OpType& op);
std::ostream& operator<<(std::ostream& os, const BufferType& buf);
std::ostream& operator<<(std::ostream& os, const Instruction& inst);
