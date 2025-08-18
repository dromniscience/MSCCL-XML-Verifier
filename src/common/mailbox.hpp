#pragma once
#include "instructions.hpp"
#include <vector>
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <memory>
#include <map>

#define MAX_TRIES 100000 // Total wait time: 100000 * 1us = 100ms
#define SLEEP_TIME std::chrono::microseconds(1)

using ChunkDataType = std::string;

struct Message {
    std::vector<ChunkDataType> chunks;
    BufferType src_buff;
    std::ptrdiff_t src_off;
    BufferType dst_buff;
    std::ptrdiff_t dst_off;
};

class Mailbox {
public:
    /**
     * @brief Sends a message to the mailbox.
     */
    void sendMessage(const Message& msg);
    /**
     * @brief Receives a message from the mailbox.
     * @return true if a message was received, false if no message was available after maxTries attempts.
     * The function will block for 1 ms between attempts.
     */
    bool receiveMessage(Message& msg);
    /**
     * @brief Checks if the mailbox is empty.
     */
    bool isEmpty() const;

private:
    std::queue<Message> inbox;
    mutable std::mutex mailboxMutex; // Protect inbox
};

class MailboxManager {
public:
    struct MapKey {
        int send_rank;
        int recv_rank;
        int chan_id;
        bool operator<(const MapKey& other) const {
            return std::tie(send_rank, recv_rank, chan_id) < std::tie(other.send_rank, other.recv_rank, other.chan_id);
        }
    };
    /**
     * Get the send mailbox for a given source and destination rank and a channel.
     * @return true if the mailbox is newly created, false if it already exists.
     */
    bool getSendMailbox(int send_rank, int recv_rank, int chan_id, std::shared_ptr<Mailbox>& mailbox);
    /**
     * Get the receive mailbox for a given source and destination rank and a channel.
     * @return true if the mailbox was created, false if it was not found after maxTries attempts.
     * The function will block for 1 ms between attempts.
     */
    bool getRecvMailbox(int send_rank, int recv_rank, int chan_id, std::shared_ptr<Mailbox>& mailbox);

    /**
     * @brief Checks if there is no pending connections.
     */
    bool checkNoPendingConnections() const;
    /**
     * @brief Checks if the channel is valid.
     *
     * Specifically, each rank cannot send to or receive from multiple ranks in a channel.
     */
    bool checkChannelLayout() const;

    /**
     * @brief Checks if there is no pending message in any mailbox.
     */
    bool checkNoPendingMessage() const;

private:
    std::map<MapKey, std::shared_ptr<Mailbox>> established_mailboxes;
    std::map<MapKey, std::shared_ptr<Mailbox>> pending_mailboxes;
    mutable std::mutex mailboxManagerMutex; // Protect mailboxes
};