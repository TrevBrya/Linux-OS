// memory_simulator_advanced.cpp
// Advanced Segmented, Paged Memory Model Simulator
// with TLB, Multi-Level Paging, LRU/FIFO, and Dynamic Segments.

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <list>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <fstream>
#include <random>
#include <chrono>

enum Protection { READ_ONLY, READ_WRITE };

// ---------- Basic Structures ----------

struct Page {
    int frame_number = -1;   // -1 if not in memory
    bool present = false;
    Protection protection = READ_WRITE;
    int last_access = 0;      // for LRU
};

struct Segment {
    int base_address = 0;
    int limit = 0;            // number of pages (0 => unused)
    Protection protection = READ_WRITE;
    int fault_count = 0;
};

// ---------- Physical Memory with FIFO / LRU ----------

class PhysicalMemory {
public:
    int num_frames;
    std::vector<bool> free_frames;
    std::queue<int> fifo_queue;      // FIFO
    std::list<int> lru_list;         // LRU
    std::map<int, int> frame_to_page; // frame -> logical id (for debugging)
    bool use_lru;

    PhysicalMemory(int frames, bool use_lru_)
        : num_frames(frames), free_frames(frames, true), use_lru(use_lru_) {}

    // allocate a frame; returns frame index
    int allocateFrame(int logical_id, int time) {
        // First try to find a free frame
        for (int i = 0; i < num_frames; ++i) {
            if (free_frames[i]) {
                free_frames[i] = false;
                frame_to_page[i] = logical_id;
                if (use_lru) {
                    lru_list.push_back(i);
                } else {
                    fifo_queue.push(i);
                }
                return i;
            }
        }

        // No free frame; need replacement
        int frame;
        if (use_lru) {
            frame = lru_list.front();
            lru_list.pop_front();
            lru_list.push_back(frame);
        } else {
            frame = fifo_queue.front();
            fifo_queue.pop();
            fifo_queue.push(frame);
        }
        frame_to_page[frame] = logical_id;
        // (Freeing the victim's page data happens in page tables logically)
        return frame;
    }

    void touchFrameLRU(int frame) {
        if (!use_lru) return;
        auto it = std::find(lru_list.begin(), lru_list.end(), frame);
        if (it != lru_list.end()) {
            lru_list.erase(it);
        }
        lru_list.push_back(frame);
    }

    // Mark frame as free (used when removing segments)
    void freeFrame(int frame) {
        if (frame < 0 || frame >= num_frames) return;
        if (free_frames[frame]) return;

        free_frames[frame] = true;
        frame_to_page.erase(frame);

        if (use_lru) {
            lru_list.remove(frame);
        } else {
            std::queue<int> tmp;
            while (!fifo_queue.empty()) {
                int f = fifo_queue.front();
                fifo_queue.pop();
                if (f != frame) tmp.push(f);
            }
            fifo_queue = std::move(tmp);
        }
    }

    double utilization() const {
        int used = std::count(free_frames.begin(), free_frames.end(), false);
        return (double)used / num_frames * 100.0;
    }
};

// ---------- Page Table & Directory Table (2-level paging) ----------

class PageTable {
public:
    std::vector<Page> pages;
    int page_size;
    int directory_index;          // which directory this belongs to

    PageTable(int numPages, int pageSize, int dirIdx)
        : pages(numPages), page_size(pageSize), directory_index(dirIdx) {}

    int getFrameNumber(int pageNum, int time, Protection accessType,
                       PhysicalMemory &physMem) {
        if (pageNum < 0 || pageNum >= (int)pages.size()) {
            throw std::runtime_error("Page Fault: Invalid page number " +
                                     std::to_string(pageNum));
        }
        Page &p = pages[pageNum];

        // Page not present -> allocate
        if (!p.present) {
            int frame = physMem.allocateFrame(pageNum, time);
            p.present = true;
            p.frame_number = frame;
            p.protection = accessType; // first time: set protection
        }

        // Protection check
        if (accessType == READ_WRITE && p.protection == READ_ONLY) {
            throw std::runtime_error(
                "Protection Violation: Cannot write to read-only page");
        }

        p.last_access = time;
        physMem.touchFrameLRU(p.frame_number);

        return p.frame_number;
    }
};

class DirectoryTable {
public:
    std::vector<PageTable *> pageTables;
    int max_pages_per_table;

    DirectoryTable(int maxPages)
        : max_pages_per_table(maxPages) {}

    ~DirectoryTable() {
        freeTables();
    }

    PageTable *getPageTable(int dirNum, int pageSize) {
        while (dirNum >= (int)pageTables.size()) {
            pageTables.push_back(
                new PageTable(max_pages_per_table, pageSize,
                              (int)pageTables.size()));
        }
        return pageTables[dirNum];
    }

    void freeTables() {
        for (auto *pt : pageTables) {
            delete pt;
        }
        pageTables.clear();
    }
};

// ---------- TLB with LRU ----------

class TLB {
private:
    std::unordered_map<std::string, int> cache; // seg:dir:page -> frame
    std::list<std::string> lruOrder;
    int maxSize;

    int hits = 0;
    int total = 0;

    std::string makeKey(int segNum, int dirNum, int pageNum) const {
        return std::to_string(segNum) + ":" +
               std::to_string(dirNum) + ":" +
               std::to_string(pageNum);
    }

public:
    TLB(int size) : maxSize(size) {}

    int get(int segNum, int dirNum, int pageNum) {
        if (maxSize <= 0) return -1;
        std::string key = makeKey(segNum, dirNum, pageNum);
        auto it = cache.find(key);
        total++;
        if (it != cache.end()) {
            hits++;
            lruOrder.remove(key);
            lruOrder.push_back(key);
            return it->second;
        }
        return -1;
    }

    void put(int segNum, int dirNum, int pageNum, int frame) {
        if (maxSize <= 0) return;
        std::string key = makeKey(segNum, dirNum, pageNum);
        if ((int)cache.size() >= maxSize) {
            // evict least recently used
            const std::string &oldKey = lruOrder.front();
            cache.erase(oldKey);
            lruOrder.pop_front();
        }
        cache[key] = frame;
        lruOrder.push_back(key);
    }

    double hitRate() const {
        return total > 0 ? (double)hits / total * 100.0 : 0.0;
    }

    void displayCache() const {
        std::cout << "TLB contents (LRU order):\n";
        for (const auto &key : lruOrder) {
            auto it = cache.find(key);
            if (it != cache.end()) {
                std::cout << "  " << key << " -> frame " << it->second << "\n";
            }
        }
        std::cout << "TLB hit rate: " << hitRate() << "%\n";
    }
};

// ---------- Segment Table & Translation ----------

class SegmentTable {
private:
    std::vector<Segment> segments;
    std::map<int, DirectoryTable *> directoryTables;
    TLB tlb;
    PhysicalMemory physMem;

    int time = 0;
    long long total_latency = 0;
    long long translation_count = 0;

public:
    SegmentTable(int tlbSize, int numFrames, bool use_lru)
        : tlb(tlbSize), physMem(numFrames, use_lru) {}

    ~SegmentTable() {
        for (auto &p : directoryTables) {
            delete p.second;
        }
    }

    void addSegment(int id, int base, int limit, Protection prot) {
        if (id < 0) {
            throw std::runtime_error("Segment id must be >= 0");
        }
        if (id >= (int)segments.size())
            segments.resize(id + 1);

        if (segments[id].limit != 0) {
            throw std::runtime_error("Segment " + std::to_string(id) +
                                     " already exists");
        }

        segments[id].base_address = base;
        segments[id].limit = limit;
        segments[id].protection = prot;
        segments[id].fault_count = 0;

        directoryTables[id] = new DirectoryTable(limit);
    }

    void removeSegment(int id) {
        if (id < 0 || id >= (int)segments.size() ||
            segments[id].limit == 0 ||
            directoryTables.find(id) == directoryTables.end()) {
            throw std::runtime_error("Cannot remove invalid segment " +
                                     std::to_string(id));
        }

        DirectoryTable *dir = directoryTables[id];
        for (auto *pt : dir->pageTables) {
            for (auto &p : pt->pages) {
                if (p.present) {
                    physMem.freeFrame(p.frame_number);
                    p.present = false;
                }
            }
        }
        dir->freeTables();
        delete dir;
        directoryTables.erase(id);

        segments[id].base_address = 0;
        segments[id].limit = 0;
        segments[id].protection = READ_WRITE;
        segments[id].fault_count = 0;
    }

    int translateAddress(int segNum, int dirNum, int pageNum,
                         int offset, Protection accessType) {
        int latency = 1 + (rand() % 10); // simulate time
        time++;
        total_latency += latency;
        translation_count++;

        if (segNum < 0 || segNum >= (int)segments.size() ||
            segments[segNum].limit == 0) {
            segments[segNum].fault_count++;
            throw std::runtime_error("Segmentation Fault: invalid segment " +
                                     std::to_string(segNum));
        }

        Segment &seg = segments[segNum];

        if (accessType == READ_WRITE && seg.protection == READ_ONLY) {
            seg.fault_count++;
            throw std::runtime_error(
                "Protection Violation: Cannot write to read-only segment");
        }

        if (pageNum < 0 || pageNum >= seg.limit) {
            seg.fault_count++;
            throw std::runtime_error("Page Fault: page " +
                                     std::to_string(pageNum) +
                                     " exceeds segment limit " +
                                     std::to_string(seg.limit));
        }

        DirectoryTable *dirTable = directoryTables[segNum];
        if (!dirTable) {
            seg.fault_count++;
            throw std::runtime_error("Segment has no directory table");
        }

        // offset bound check (assume page_size constant across tables)
        PageTable *tmpPT = dirTable->getPageTable(0, 1000);
        int page_size = tmpPT->page_size;
        if (offset < 0 || offset >= page_size) {
            seg.fault_count++;
            throw std::runtime_error("Offset Fault: offset " +
                                     std::to_string(offset) +
                                     " exceeds page size " +
                                     std::to_string(page_size));
        }

        // TLB lookup
        int frame = tlb.get(segNum, dirNum, pageNum);
        if (frame != -1) {
            int physical = seg.base_address + frame * page_size + offset;
            return physical;
        }

        // Page table lookup
        PageTable *pageTable = dirTable->getPageTable(dirNum, page_size);
        frame = pageTable->getFrameNumber(pageNum, time, accessType, physMem);
        tlb.put(segNum, dirNum, pageNum, frame);

        int physical = seg.base_address + frame * page_size + offset;
        return physical;
    }

    void displayStats() const {
        std::cout << "\n=== Page Fault Statistics ===\n";
        for (size_t i = 0; i < segments.size(); ++i) {
            if (segments[i].limit == 0) continue;
            std::cout << "Segment " << i
                      << ": Faults = " << segments[i].fault_count;
            if (translation_count > 0) {
                double frac = (double)segments[i].fault_count /
                              translation_count;
                if (frac > 0.2) {
                    std::cout << "  (Suggestion: consider increasing limit)";
                }
            }
            std::cout << "\n";
        }
        std::cout << "Average translation latency: ";
        if (translation_count > 0) {
            std::cout << (double)total_latency / translation_count << "\n";
        } else {
            std::cout << "N/A\n";
        }
        std::cout << "Physical memory utilization: "
                  << physMem.utilization() << "%\n";
        tlb.displayCache();
        std::cout << "==============================\n";
    }

    void printMemoryMap() const {
        std::cout << "\n=== Memory Map ===\n";
        for (size_t i = 0; i < segments.size(); ++i) {
            if (segments[i].limit == 0) continue;
            std::cout << "Segment " << i
                      << " : Base=" << segments[i].base_address
                      << "  Limit=" << segments[i].limit
                      << "  Prot=" << (segments[i].protection == READ_ONLY ? "RO" : "RW")
                      << "  Faults=" << segments[i].fault_count << "\n";
        }
        std::cout << "===================\n";
    }

    // helpers for random generator / external callers
    int numSegments() const { return (int)segments.size(); }
    const Segment &getSegment(int i) const { return segments[i]; }
};

// ---------- Batch processing & random tests ----------

void processBatchFile(SegmentTable &st, const std::string &filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open batch file: " + filename);
    }
    std::ofstream log("batch_results.txt");
    int faults = 0;
    int translations = 0;

    int segNum, dirNum, pageNum, offset, access;
    while (file >> segNum >> dirNum >> pageNum >> offset >> access) {
        try {
            Protection acc = access ? READ_WRITE : READ_ONLY;
            int addr = st.translateAddress(segNum, dirNum, pageNum, offset, acc);
            log << "Address (" << segNum << "," << dirNum << ","
                << pageNum << "," << offset << "," << access
                << "): Physical=" << addr << "\n";
        } catch (const std::runtime_error &e) {
            log << "Address (" << segNum << "," << dirNum << ","
                << pageNum << "," << offset << "," << access
                << "): ERROR: " << e.what() << "\n";
            faults++;
        }
        translations++;
    }

    log << "Fault rate: "
        << (translations > 0 ? (double)faults / translations * 100.0 : 0.0)
        << "%\n";
    std::cout << "Batch processing complete. See batch_results.txt\n";
}

void generateRandomAddresses(SegmentTable &st, int num,
                             double validRatio,
                             const std::string &logfile) {
    std::ofstream log(logfile);
    if (!log) {
        throw std::runtime_error("Cannot open log file for random output");
    }

    std::mt19937 gen(
        (unsigned)std::chrono::system_clock::now().time_since_epoch().count());
    int faults = 0;

    for (int i = 0; i < num; ++i) {
        bool valid = ((double)gen() / gen.max()) < validRatio;

        int segNum, dirNum, pageNum, offset;
        Protection access;

        if (valid && st.numSegments() > 0) {
            // choose a random existing segment
            int tryCount = 0;
            do {
                segNum = gen() % st.numSegments();
                tryCount++;
            } while ((st.getSegment(segNum).limit == 0) &&
                     tryCount < 20);

            const Segment &seg = st.getSegment(segNum);
            if (seg.limit == 0) { // fallback invalid
                valid = false;
            } else {
                pageNum = gen() % seg.limit;
                dirNum = gen() % 4;          // arbitrary small directory range
                offset = gen() % 1000;       // same page size as earlier
                access = (gen() % 2) ? READ_WRITE : READ_ONLY;
            }
        }

        if (!valid) {
            // intentionally invalid: random garbage
            segNum = gen() % (st.numSegments() + 3) - 1;
            dirNum = gen() % 8;
            pageNum = gen() % 20;
            offset = gen() % 2000;
            access = (gen() % 2) ? READ_WRITE : READ_ONLY;
        }

        try {
            int addr = st.translateAddress(segNum, dirNum, pageNum, offset,
                                           access);
            log << "Address (" << segNum << "," << dirNum << ","
                << pageNum << "," << offset << ","
                << (access == READ_ONLY ? "Read" : "Write")
                << "): Physical=" << addr << "\n";
        } catch (const std::runtime_error &e) {
            log << "Address (" << segNum << "," << dirNum << ","
                << pageNum << "," << offset << ","
                << (access == READ_ONLY ? "Read" : "Write")
                << "): ERROR: " << e.what() << "\n";
            faults++;
        }
    }

    log << "Page fault rate: "
        << (num > 0 ? (double)faults / num * 100.0 : 0.0) << "%\n";
    std::cout << "Random address test complete. See " << logfile << "\n";
}

// ---------- main ----------

int main(int argc, char *argv[]) {
    srand((unsigned)time(nullptr));

    int numFrames = 16;
    int tlbSize = 4;
    int pageSize = 1000;  // fixed for this lab
    bool use_lru = true;
    std::string batchFile;

    // Simple command-line parsing: flag value pairs
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "--frames") {
            numFrames = std::stoi(argv[i + 1]);
        } else if (arg == "--tlbsize") {
            tlbSize = std::stoi(argv[i + 1]);
        } else if (arg == "--pagesize") {
            pageSize = std::stoi(argv[i + 1]); // stored indirectly
        } else if (arg == "--replace") {
            std::string pol = argv[i + 1];
            use_lru = (pol == "lru" || pol == "LRU");
        } else if (arg == "--batch") {
            batchFile = argv[i + 1];
        }
    }

    SegmentTable segmentTable(tlbSize, numFrames, use_lru);

    // Add two starter segments as in the description
    segmentTable.addSegment(0, 0, 5, READ_WRITE);
    segmentTable.addSegment(1, 5000, 3, READ_ONLY);

    if (!batchFile.empty()) {
        try {
            processBatchFile(segmentTable, batchFile);
            segmentTable.displayStats();
        } catch (const std::runtime_error &e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
        return 0;
    }

    // Interactive mode
    std::cout << "Advanced Memory Simulator\n";
    std::cout << "Commands:\n"
              << "  add <id> <base> <limit> <prot(RO/RW)>\n"
              << "  remove <id>\n"
              << "  translate <seg> <dir> <page> <offset> <access(0=R,1=W)>\n"
              << "  random <num>\n"
              << "  stats\n"
              << "  quit\n";

    segmentTable.printMemoryMap();

    std::string command;
    while (true) {
        std::cout << "\n> ";
        if (!(std::cin >> command)) break;

        try {
            if (command == "add") {
                int id, base, limit;
                std::string protStr;
                std::cin >> id >> base >> limit >> protStr;
                Protection prot =
                    (protStr == "RO" ? READ_ONLY : READ_WRITE);
                segmentTable.addSegment(id, base, limit, prot);
                std::cout << "Segment " << id << " added.\n";
            } else if (command == "remove") {
                int id;
                std::cin >> id;
                segmentTable.removeSegment(id);
                std::cout << "Segment " << id << " removed.\n";
            } else if (command == "translate") {
                int segNum, dirNum, pageNum, offset, accessInt;
                std::cin >> segNum >> dirNum >> pageNum >> offset >> accessInt;
                Protection access =
                    accessInt ? READ_WRITE : READ_ONLY;
                int addr = segmentTable.translateAddress(
                    segNum, dirNum, pageNum, offset, access);
                std::cout << "Physical address: " << addr << "\n";
            } else if (command == "random") {
                int num;
                std::cin >> num;
                generateRandomAddresses(segmentTable, num, 0.7,
                                        "random_results.txt");
            } else if (command == "stats") {
                segmentTable.displayStats();
            } else if (command == "quit") {
                break;
            } else {
                std::cout << "Unknown command.\n";
            }
        } catch (const std::runtime_error &e) {
            std::cerr << "Error: " << e.what() << "\n";
        }
        segmentTable.printMemoryMap();
    }

    std::cout << "Exiting simulator.\n";
    return 0;
}
