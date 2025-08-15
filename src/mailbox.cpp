#include "mailbox.hpp"
#include <set>

void Mailbox::sendMessage(const Message& msg) {
    std::lock_guard<std::mutex> lock(mailboxMutex);
    inbox.push(msg);
}

bool Mailbox::receiveMessage(Message& msg) {
    for (int tries = 0; tries < MAX_TRIES; ++tries) {
        {
            std::lock_guard<std::mutex> lock(mailboxMutex);
            if (!inbox.empty()) {
                msg = inbox.front();
                inbox.pop();
                return true;
            }
        }
        std::this_thread::sleep_for(SLEEP_TIME);
    }
    return false;
}

bool Mailbox::isEmpty() const {
    std::lock_guard<std::mutex> lock(mailboxMutex);
    return inbox.empty();
}

bool MailboxManager::getSendMailbox(int send_rank, int recv_rank, int chan_id, std::shared_ptr<Mailbox>& mailbox) {
    MapKey key{send_rank, recv_rank, chan_id};
    std::lock_guard<std::mutex> lock(mailboxManagerMutex);
    auto it = established_mailboxes.find(key);
    if (it != established_mailboxes.end()) {
        mailbox = it->second;
        return false;
    } else {
        mailbox = std::make_shared<Mailbox>();
        pending_mailboxes[key] = mailbox;
        return true;
    }
}

bool MailboxManager::getRecvMailbox(int send_rank, int recv_rank, int chan_id, std::shared_ptr<Mailbox>& mailbox) {
    MapKey key{send_rank, recv_rank, chan_id};
    for (int tries = 0; tries < MAX_TRIES; ++tries) {
        {
            std::lock_guard<std::mutex> lock(mailboxManagerMutex);
            auto it = pending_mailboxes.find(key);
            if (it != pending_mailboxes.end()) {
                mailbox = it->second;
                established_mailboxes[key] = mailbox;
                pending_mailboxes.erase(it);
                return true;
            }
        }
        std::this_thread::sleep_for(SLEEP_TIME);
    }
    return false;
}

bool MailboxManager::checkNoPendingConnections() const {
    std::lock_guard<std::mutex> lock(mailboxManagerMutex);
    return pending_mailboxes.empty();
}

bool MailboxManager::checkChannelLayout() const {
    std::map<int, std::set<int>> chan_send; // chan_id -> set of send_ranks
    std::map<int, std::set<int>> chan_recv; // chan_id -> set of recv_ranks
    std::lock_guard<std::mutex> lock(mailboxManagerMutex);
    for (const auto& [key, mailbox] : established_mailboxes) {
        auto& senders_set = chan_send[key.chan_id];
        auto& receivers_set = chan_recv[key.chan_id];
        if (senders_set.count(key.send_rank) > 0) {
            return false; // Multiple sends from the same rank in a channel
        }
        if (receivers_set.count(key.recv_rank) > 0) {
            return false; // Multiple receives to the same rank in a channel
        }
        senders_set.insert(key.send_rank);
        receivers_set.insert(key.recv_rank);
    }
    return true;
}

bool MailboxManager::checkNoPendingMessage() const {
    std::lock_guard<std::mutex> lock(mailboxManagerMutex);
    for (const auto& [key, mailbox] : established_mailboxes) {
        if (!mailbox->isEmpty()) {
            return false; // Found a mailbox with pending messages
        }
    }
    return true;
}