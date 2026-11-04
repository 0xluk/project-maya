// glm_link: see strata/core/glm_link.hpp
#include "strata/core/glm_link.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#include <vector>
#endif

namespace strata::core::glmlink {
namespace {

int g_data = -1, g_ctl = -1, g_listen = -1;
std::atomic<bool> g_up{false};
// what open() was given (reopen / reaccept HELLO with it again)
int g_split = 0, g_my_chunk = 0, g_peer_chunk = -1;
int64_t g_ctx = 0;
// the request in flight: the back saw the front's kEnd (a cancel) - its position and flags
bool g_cancel = false;
int64_t g_end_pos = -1;
int32_t g_end_flags = 0;
bool g_bf16 = true;
bool g_pack = true;   // STRATA_GLM_LINK_PACK: the bf16 rows' high bytes nibble-coded (lossless)
std::mutex g_ctl_mu;
// stats
uint64_t s_hops = 0, s_hop_bytes = 0, s_toks = 0;
double s_send_ms = 0, s_recv_ms = 0;

const char* env2(const char* a, const char* b) {
    const char* v = std::getenv(a);
    return v != nullptr && v[0] ? v : std::getenv(b);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void fail(std::string& err, const std::string& what) {
    if (g_up.exchange(false)) std::fprintf(stderr, "glm link: DOWN - %s\n", what.c_str());
    err = "glm link: " + what;
}

#if !defined(_WIN32)
bool write_all(int fd, const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
        const ssize_t w = ::send(fd, c, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        c += w;
        n -= (size_t) w;
    }
    return true;
}

bool read_all(int fd, void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const ssize_t r = ::recv(fd, c, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        c += r;
        n -= (size_t) r;
    }
    return true;
}

// header and payload in one call (TCP_NODELAY: a lone header would leave as its own segment)
bool send_msg(const Hdr& h, const void* data, size_t bytes) {
    struct iovec iov[2];
    iov[0].iov_base = (void*) &h;
    iov[0].iov_len = sizeof h;
    iov[1].iov_base = (void*) data;
    iov[1].iov_len = bytes;
    size_t total = sizeof h + bytes, done = 0;
    int i0 = 0;
    while (done < total) {
        struct msghdr m {};
        m.msg_iov = iov + i0;
        m.msg_iovlen = (size_t) (bytes > 0 ? 2 - i0 : 1);
        const ssize_t w = ::sendmsg(g_data, &m, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        done += (size_t) w;
        size_t adv = (size_t) w;
        while (adv > 0 && i0 < 2) {
            if (adv >= iov[i0].iov_len) {
                adv -= iov[i0].iov_len;
                iov[i0].iov_len = 0;
                ++i0;
            } else {
                iov[i0].iov_base = (char*) iov[i0].iov_base + adv;
                iov[i0].iov_len -= adv;
                adv = 0;
            }
        }
    }
    return true;
}

void tune(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    int buf = 8 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
}

bool split_hostport(const std::string& s, std::string& host, std::string& port) {
    const size_t c = s.rfind(':');
    if (c == std::string::npos || c + 1 >= s.size()) return false;
    host = s.substr(0, c);
    port = s.substr(c + 1);
    if (host.empty()) host = "0.0.0.0";
    return true;
}

int connect_once(const std::string& host, const std::string& port) {
    struct addrinfo hints {}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) return -1;
    int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd >= 0 && ::connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        ::close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd >= 0) tune(fd);
    return fd;
}
#endif

}  // namespace

Role role() {
    static const Role r = [] {
        const char* v = env2("STRATA_GLM_ROLE", "MAYA_GLM_ROLE");
        if (v == nullptr || !v[0]) return kNone;
        if (std::strcmp(v, "front") == 0) return kFront;
        if (std::strcmp(v, "back") == 0) return kBack;
        std::fprintf(stderr, "glm link: STRATA_GLM_ROLE=%s is not front|back - ignored\n", v);
        return kNone;
    }();
    return r;
}

const char* role_name() { return role() == kFront ? "front" : role() == kBack ? "back" : "none"; }
bool up() { return g_up.load(); }
int ctl_fd() { return role() == kBack ? g_ctl : -1; }

#if defined(_WIN32)
bool open(int, int64_t, int, int&, std::string& err) {
    err = "glm link: not on Windows";
    return false;
}
bool send_hop(int64_t, int, const void*, size_t, std::string& err) { err = "glm link: not on Windows"; return false; }
bool recv_hop(int64_t, int, void*, size_t, std::string& err) { err = "glm link: not on Windows"; return false; }
bool send_tok(Type, int64_t, int32_t) { return false; }
void abort_link(const std::string&) {}
bool recv_tok(Type, int64_t, int32_t&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool send_pair(Type, int64_t, int32_t, int32_t) { return false; }
bool recv_pair(Type, int64_t, int32_t&, int32_t&, std::string& err) { err = "glm link: not on Windows"; return false; }
void ctl_send_line(const std::string&) {}
void ctl_close() {}
void report() {}
void request_begin() {}
bool cancelled() { return false; }
bool begin_exchange(int64_t, int32_t, int64_t&, int32_t&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool end_front(int64_t, int32_t, bool&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool end_back(int64_t, int32_t&, bool&, std::string& err) { err = "glm link: not on Windows"; return false; }
bool bf16() { return false; }
bool idle_check() { return false; }
bool reopen(std::string& err) { err = "glm link: not on Windows"; return false; }
void wake() {}
bool reaccept(std::string& err) { err = "glm link: not on Windows"; return false; }
#else
// this side's settings both must share: the hop's format and the prompt's chunking floor (the chunks must match)
static std::string link_config() {
    const char* mc = std::getenv("STRATA_GLM_PREFILL_MIN_CHUNKS");
    if (mc == nullptr || !mc[0]) mc = std::getenv("MAYA_GLM_PREFILL_MIN_CHUNKS");
    return std::string("bf16=") + (g_bf16 ? "1" : "0") + " pack=" + (g_bf16 && g_pack ? "1" : "0") +
           " min_chunks=" + (mc != nullptr ? mc : "");
}

static bool hello(std::string& err) {
    const std::string cfg = link_config();
    Hdr h{kMagic, kHello, kVersion, g_split, g_my_chunk, (uint64_t) g_ctx}, p{};
    // (the HELLO's reply within 60 s: a back that accepted but hangs must not hold the front forever)
    struct timeval tv {60, 0}, t0v {0, 0};
    setsockopt(g_data, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    const uint32_t cl = (uint32_t) cfg.size();
    bool ok = send_msg(h, nullptr, 0) && write_all(g_data, &cl, 4) && write_all(g_data, cfg.data(), cl) &&
              read_all(g_data, &p, sizeof p);
    uint32_t pl = 0;
    std::string pcfg;
    if (ok && p.magic == kMagic && p.type == kHello && p.pos == kVersion) {
        ok = read_all(g_data, &pl, 4) && pl < 4096;
        if (ok) {
            pcfg.assign(pl, '\0');
            ok = pl == 0 || read_all(g_data, &pcfg[0], pl);
        }
    }
    setsockopt(g_data, SOL_SOCKET, SO_RCVTIMEO, &t0v, sizeof t0v);
    if (!ok) {
        err = "glm link: HELLO failed";
        return false;
    }
    if (p.magic != kMagic || p.type != kHello || p.pos != kVersion) {
        err = "glm link: the other side speaks another protocol";
        return false;
    }
    if (p.n != g_split || (int64_t) p.bytes != g_ctx) {
        err = "glm link: the other side splits at " + std::to_string(p.n) + " with context " +
              std::to_string((long long) p.bytes) + ", this one at " + std::to_string(g_split) + " with " +
              std::to_string((long long) g_ctx);
        return false;
    }
    if (pcfg != cfg) {
        err = "glm link: the other side runs with '" + pcfg + "', this one with '" + cfg + "'";
        return false;
    }
    if (g_peer_chunk >= 0 && p.val != g_peer_chunk) {
        err = "glm link: the other side's prompt chunk is " + std::to_string(p.val) + " now, " +
              std::to_string(g_peer_chunk) + " before";
        return false;
    }
    g_peer_chunk = p.val;
    return true;
}

static void close_fds() {
    if (g_data >= 0) ::close(g_data);
    if (g_ctl >= 0) ::close(g_ctl);
    g_data = g_ctl = -1;
}

// the front: both connections (DATA, CTRL), retrying for wait_s seconds
static bool connect_pair(const std::string& host, const std::string& port, double wait_s, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int which = 0; which < 2; ++which) {
        int fd = -1;
        while ((fd = connect_once(host, port)) < 0) {
            if (ms_since(t0) > wait_s * 1000.0) {
                close_fds();
                err = "glm link: no back at " + host + ":" + port + " after " + std::to_string((int) wait_s) + " s";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        const char* tag = which == 0 ? "DATA" : "CTRL";
        if (!write_all(fd, tag, 4)) {
            ::close(fd);
            close_fds();
            err = "glm link: the connection's tag did not go out";
            return false;
        }
        (which == 0 ? g_data : g_ctl) = fd;
    }
    return true;
}

// the back: the front's two connections on the listening socket
static bool accept_pair(std::string& err) {
    while (g_data < 0 || g_ctl < 0) {
        const int fd = ::accept(g_listen, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) continue;
            err = std::string("glm link: accept: ") + std::strerror(errno);
            return false;
        }
        tune(fd);
        char tag[4];
        struct timeval tv {10, 0}, t0v {0, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        if (!read_all(fd, tag, 4)) { ::close(fd); continue; }
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t0v, sizeof t0v);
        if (std::memcmp(tag, "DATA", 4) == 0) {
            if (g_data >= 0) ::close(g_data);   // (a front that started over: its newest pair stands)
            g_data = fd;
        } else if (std::memcmp(tag, "CTRL", 4) == 0) {
            if (g_ctl >= 0) ::close(g_ctl);
            g_ctl = fd;
        } else {
            ::close(fd);
        }
    }
    return true;
}

static bool link_hostport(std::string& host, std::string& port, std::string& err) {
    const char* lk = env2("STRATA_GLM_LINK", "MAYA_GLM_LINK");
    if (lk == nullptr || !split_hostport(lk, host, port)) {
        err = "glm link: STRATA_GLM_LINK=host:port (or --link) is needed with --role";
        return false;
    }
    return true;
}

void set_split(int split_at) { g_split = split_at; }
int split_at() { return g_split; }

bool open(int split_at, int64_t max_ctx, int my_chunk, int& peer_chunk, std::string& err) {
    std::string host, port;
    if (!link_hostport(host, port, err)) return false;
    g_split = split_at;
    g_ctx = max_ctx;
    g_my_chunk = my_chunk;
    if (const char* b = env2("STRATA_GLM_LINK_BF16", "MAYA_GLM_LINK_BF16")) g_bf16 = std::atoi(b) != 0;
    if (const char* b = env2("STRATA_GLM_LINK_PACK", "MAYA_GLM_LINK_PACK")) g_pack = std::atoi(b) != 0;
    if (!g_bf16) g_pack = false;
    const auto t0 = std::chrono::steady_clock::now();
    if (role() == kFront) {
        double wait_s = 1800;
        if (const char* w = std::getenv("STRATA_GLM_LINK_WAIT")) wait_s = std::atof(w);
        std::fprintf(stderr, "glm link: front connecting to %s:%s\n", host.c_str(), port.c_str());
        if (!connect_pair(host, port, wait_s, err)) return false;
    } else {
        struct addrinfo hints {}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || res == nullptr) {
            err = "glm link: cannot resolve " + host;
            return false;
        }
        const int ls = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (ls < 0 || ::bind(ls, res->ai_addr, res->ai_addrlen) != 0 || ::listen(ls, 4) != 0) {
            freeaddrinfo(res);
            err = std::string("glm link: cannot listen on ") + host + ":" + port + ": " + std::strerror(errno);
            return false;
        }
        freeaddrinfo(res);
        g_listen = ls;   // (kept: the next front is accepted on it - reaccept)
        std::fprintf(stderr, "glm link: back listening on %s:%s\n", host.c_str(), port.c_str());
        if (!accept_pair(err)) return false;
    }
    if (!hello(err)) return false;
    peer_chunk = g_peer_chunk;
    g_up.store(true);
    std::fprintf(stderr, "glm link: %s up in %.1f s (boundary layer %d, context %lld, prompt chunks %d / %d, %s)\n",
                 role_name(), ms_since(t0) / 1000.0, split_at, (long long) max_ctx, my_chunk, peer_chunk,
                 link_config().c_str());
    std::atexit(report);
    return true;
}

bool bf16() { return g_bf16; }

bool idle_check() {
    if (!up() || g_data < 0) return false;
    struct pollfd pf {g_data, POLLIN | POLLRDHUP, 0};
    // between requests nothing is due from the back: anything readable is its end (EOF) or an ERR
    if (::poll(&pf, 1, 0) > 0 && (pf.revents & (POLLIN | POLLRDHUP | POLLHUP | POLLERR))) {
        std::string err;
        fail(err, "the back has gone");
        return false;
    }
    return true;
}

bool reopen(std::string& err) {
    std::string host, port;
    if (!link_hostport(host, port, err)) return false;
    g_up.store(false);
    {
        std::lock_guard<std::mutex> lk(g_ctl_mu);
        close_fds();
        double wait_s = 5;
        if (const char* w = std::getenv("STRATA_GLM_RELINK_WAIT")) wait_s = std::atof(w);
        const auto t0 = std::chrono::steady_clock::now();
        if (!connect_pair(host, port, wait_s, err)) return false;
        if (!hello(err)) {
            close_fds();
            return false;
        }
        std::fprintf(stderr, "glm link: front connected again in %.1f s\n", ms_since(t0) / 1000.0);
    }
    g_up.store(true);
    return true;
}

void wake() {
    g_up.store(false);
    if (g_data >= 0) ::shutdown(g_data, SHUT_RDWR);
}

bool reaccept(std::string& err) {
    g_up.store(false);
    {
        std::lock_guard<std::mutex> lk(g_ctl_mu);
        close_fds();
    }
    if (g_listen < 0) {
        err = "glm link: not listening";
        return false;
    }
    std::fprintf(stderr, "glm link: the front has gone - waiting for it again\n");
    for (;;) {
        if (!accept_pair(err)) return false;
        if (hello(err)) break;
        std::fprintf(stderr, "%s - waiting for another front\n", err.c_str());
        close_fds();
    }
    g_up.store(true);
    std::fprintf(stderr, "glm link: back up again\n");
    return true;
}

// THE BF16 HOP: the rows cross as bf16 (the top half of each fp32, rounded to nearest even - ggml's rounding), in
// blocks of kBlk values staged per thread, so the conversion of one block overlaps the wire of the one before
constexpr size_t kBlk = 1 << 20;
static inline uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// THE PACKED HOP (STRATA_GLM_LINK_PACK=1, default with bf16; lossless): a bf16 value's high byte (sign and 7 exponent
// bits) takes few values in a residual row (entropy ~3.2 bits, the 15 commonest ~98.5%), so each block sends its low
// bytes as they are, the high bytes as 4-bit codes into the block's table of its 15 commonest (15: escape, the byte
// follows in the escape list) - ~12.1 bits a value instead of 16.  A block: u32 payload bytes, u32 values, u32 escapes,
// u8 table[16], then low bytes, nibbles, escapes.
struct PackHdr {
    uint32_t len, k, n_esc;
    uint8_t table[16];
};
static size_t pack_block(const float* src, size_t k, std::vector<uint8_t>& out) {
    thread_local std::vector<uint16_t> b;
    b.resize(k);
    uint32_t hist[256] = {0};
    for (size_t j = 0; j < k; ++j) {
        b[j] = f2bf(src[j]);
        ++hist[b[j] >> 8];
    }
    PackHdr h{};
    uint8_t code[256];
    std::memset(code, 15, sizeof code);
    for (int t = 0; t < 15; ++t) {   // the 15 commonest high bytes
        int best = -1;
        for (int v = 0; v < 256; ++v)
            if (code[v] == 15 && hist[v] > 0 && (best < 0 || hist[v] > hist[best])) best = v;
        if (best < 0) break;
        code[best] = (uint8_t) t;
        h.table[t] = (uint8_t) best;
    }
    size_t n_esc = 0;
    for (int v = 0; v < 256; ++v)
        if (code[v] == 15) n_esc += hist[v];
    const size_t nb = (k + 1) / 2, total = sizeof(PackHdr) + k + nb + n_esc;
    out.resize(total);
    uint8_t* lo = out.data() + sizeof(PackHdr);
    uint8_t* nib = lo + k;
    uint8_t* esc = nib + nb;
    std::memset(nib, 0, nb);
    size_t e = 0;
    for (size_t j = 0; j < k; ++j) {
        const uint8_t hi = (uint8_t) (b[j] >> 8), c = code[hi];
        lo[j] = (uint8_t) b[j];
        nib[j >> 1] |= (uint8_t) (c << ((j & 1) * 4));
        if (c == 15) esc[e++] = hi;
    }
    h.len = (uint32_t) (total - 4);
    h.k = (uint32_t) k;
    h.n_esc = (uint32_t) n_esc;
    std::memcpy(out.data(), &h, sizeof h);
    return total;
}
static bool unpack_block(const uint8_t* p, size_t k_want, uint32_t* dst) {
    PackHdr h;
    std::memcpy(&h, p, sizeof h);
    if (h.k != k_want) return false;
    const uint8_t* lo = p + sizeof(PackHdr);
    const uint8_t* nib = lo + h.k;
    const uint8_t* esc = nib + (h.k + 1) / 2;
    size_t e = 0;
    for (size_t j = 0; j < h.k; ++j) {
        const uint8_t c = (uint8_t) ((nib[j >> 1] >> ((j & 1) * 4)) & 15);
        const uint8_t hi = c == 15 ? (e < h.n_esc ? esc[e++] : 0) : h.table[c];
        dst[j] = ((uint32_t) hi << 24) | ((uint32_t) lo[j] << 16);
    }
    return e == h.n_esc;
}

bool send_hop(int64_t pos, int n, const void* data, size_t bytes, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    bool ok;
    size_t wire_total = 0;
    if (!g_bf16) {
        Hdr h{kMagic, kHop, pos, n, 0, (uint64_t) bytes};
        ok = send_msg(h, data, bytes);
    } else if (g_pack) {
        // (the header carries the values, not the bytes: the blocks say their own length)
        thread_local std::vector<uint8_t> st;
        const size_t nv = bytes / sizeof(float);
        const float* src = (const float*) data;
        Hdr h{kMagic, kHop, pos, n, 2, (uint64_t) nv};
        ok = true;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            const size_t w = pack_block(src + i, k, st);
            wire_total += w;
            ok = i == 0 ? send_msg(h, st.data(), w) : write_all(g_data, st.data(), w);
        }
        if (nv == 0) ok = send_msg(h, nullptr, 0);
    } else {
        thread_local std::vector<uint16_t> st;
        const size_t nv = bytes / sizeof(float);
        const float* src = (const float*) data;
        Hdr h{kMagic, kHop, pos, n, 1, (uint64_t) (nv * 2)};
        st.resize(std::min(nv, kBlk));
        ok = true;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            for (size_t j = 0; j < k; ++j) st[j] = f2bf(src[i + j]);
            ok = i == 0 ? send_msg(h, st.data(), k * 2) : write_all(g_data, st.data(), k * 2);
        }
        if (nv == 0) ok = send_msg(h, nullptr, 0);
    }
    if (!ok) {
        fail(err, "HOP send failed");
        return false;
    }
    s_send_ms += ms_since(t0);
    ++s_hops;
    s_hop_bytes += g_pack ? wire_total : g_bf16 ? bytes / 2 : bytes;
    return true;
}

// read one header; an ERR from the other side becomes this side's error
static bool recv_hdr(Hdr& h, uint32_t want, std::string& err) {
    if (!read_all(g_data, &h, sizeof h) || h.magic != kMagic) {
        fail(err, "the connection is lost");
        return false;
    }
    if (h.type == kEnd && want != kEnd && role() == kBack) {
        // THE FRONT ENDED THE REQUEST (a cancel): this side stops where it is, the link stays up
        g_cancel = true;
        g_end_pos = h.pos;
        g_end_flags = h.val;
        err = "cancelled";
        return false;
    }
    if (h.type == kErr) {
        std::string why((size_t) h.bytes, '\0');
        if (h.bytes > 0 && !read_all(g_data, &why[0], (size_t) h.bytes)) why = "?";
        fail(err, "the other side failed: " + why);
        return false;
    }
    if (h.type != want) {
        fail(err, "expected message " + std::to_string(want) + ", got " + std::to_string(h.type) + " (the sides are out of step)");
        return false;
    }
    return true;
}

bool recv_hop(int64_t pos, int n, void* dst, size_t bytes, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    Hdr h{};
    if (!recv_hdr(h, kHop, err)) return false;
    size_t wire = g_pack ? bytes / 4 : g_bf16 ? bytes / 2 : bytes;   // (packed: the header carries the values)
    if (h.pos != pos || h.n != n || h.bytes != wire || h.val != (g_pack ? 2 : g_bf16 ? 1 : 0)) {
        fail(err, "HOP for positions " + std::to_string((long long) h.pos) + "+" + std::to_string(h.n) +
                      ", expected " + std::to_string((long long) pos) + "+" + std::to_string(n) + " (out of step)");
        return false;
    }
    bool ok;
    if (!g_bf16) {
        ok = read_all(g_data, dst, bytes);
    } else if (g_pack) {
        thread_local std::vector<uint8_t> st;
        const size_t nv = bytes / sizeof(float);
        uint32_t* out = (uint32_t*) dst;
        ok = true;
        wire = 0;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            uint32_t len = 0;
            ok = read_all(g_data, &len, 4) && len >= sizeof(PackHdr) - 4 && len <= 4 * kBlk;
            if (!ok) break;
            st.resize(len + 4);
            std::memcpy(st.data(), &len, 4);
            ok = read_all(g_data, st.data() + 4, len) && unpack_block(st.data(), k, out + i);
            wire += len + 4;
        }
    } else {
        thread_local std::vector<uint16_t> st;
        const size_t nv = bytes / sizeof(float);
        uint32_t* out = (uint32_t*) dst;
        st.resize(std::min(nv, kBlk));
        ok = true;
        for (size_t i = 0; ok && i < nv; i += kBlk) {
            const size_t k = std::min(kBlk, nv - i);
            ok = read_all(g_data, st.data(), k * 2);
            for (size_t j = 0; ok && j < k; ++j) out[i + j] = (uint32_t) st[j] << 16;
        }
    }
    if (!ok) {
        fail(err, "the connection is lost in a HOP");
        return false;
    }
    s_recv_ms += ms_since(t0);
    ++s_hops;
    s_hop_bytes += wire;
    return true;
}

bool send_tok(Type t, int64_t pos, int32_t tok) {
    if (!up()) return false;
    Hdr h{kMagic, (uint32_t) t, pos, 0, tok, 0};
    if (!send_msg(h, nullptr, 0)) {
        std::string err;
        fail(err, "TOK send failed");
        return false;
    }
    ++s_toks;
    return true;
}

void abort_link(const std::string& why) {
    if (!up() || g_data < 0) return;
    Hdr e{kMagic, kErr, 0, 0, 0, (uint64_t) why.size()};
    send_msg(e, why.data(), why.size());
    std::string err;
    fail(err, "aborted: " + why);
    ::shutdown(g_data, SHUT_RDWR);
}

bool recv_tok(Type t, int64_t pos, int32_t& tok, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    Hdr h{};
    if (!recv_hdr(h, t, err)) return false;
    if (h.pos != pos) {
        fail(err, "token for position " + std::to_string((long long) h.pos) + ", expected " +
                      std::to_string((long long) pos) + " (out of step)");
        return false;
    }
    tok = h.val;
    s_recv_ms += ms_since(t0);
    ++s_toks;
    return true;
}

bool send_pair(Type t, int64_t pos, int32_t val, int32_t n) {
    if (!up()) return false;
    Hdr h{kMagic, (uint32_t) t, pos, n, val, 0};
    if (!send_msg(h, nullptr, 0)) {
        std::string err;
        fail(err, "message send failed");
        return false;
    }
    ++s_toks;
    return true;
}

bool recv_pair(Type t, int64_t pos, int32_t& val, int32_t& n, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    Hdr h{};
    if (!recv_hdr(h, t, err)) return false;
    if (h.pos != pos) {
        fail(err, "message " + std::to_string(t) + " for position " + std::to_string((long long) h.pos) + ", expected " +
                      std::to_string((long long) pos) + " (out of step)");
        return false;
    }
    val = h.val;
    n = h.n;
    s_recv_ms += ms_since(t0);
    ++s_toks;
    return true;
}

void request_begin() {
    g_cancel = false;
    g_end_pos = -1;
    g_end_flags = 0;
}

bool cancelled() { return g_cancel; }

bool begin_exchange(int64_t reuse, int32_t mode, int64_t& peer_reuse, int32_t& peer_mode, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    Hdr h{kMagic, kBegin, reuse, 0, mode, 0}, p{};
    if (role() == kFront) {
        if (!send_msg(h, nullptr, 0)) { fail(err, "BEGIN send failed"); return false; }
        if (!recv_hdr(p, kBegin, err)) return false;
    } else {
        if (!recv_hdr(p, kBegin, err)) return false;
        if (!send_msg(h, nullptr, 0)) { fail(err, "BEGIN send failed"); return false; }
    }
    peer_reuse = p.pos;
    peer_mode = p.val;
    return true;
}

// read and drop messages until one of type t (its header in h); an ERR or a lost connection fails
static bool drain_until(uint32_t t, Hdr& h, int& dropped, std::string& err) {
    dropped = 0;
    std::vector<char> junk;
    for (;;) {
        if (!read_all(g_data, &h, sizeof h) || h.magic != kMagic) {
            fail(err, "the connection is lost");
            return false;
        }
        if (h.type == t) return true;
        if (h.type == kErr) {
            std::string why((size_t) h.bytes, '\0');
            if (h.bytes > 0 && !read_all(g_data, &why[0], (size_t) h.bytes)) why = "?";
            fail(err, "the other side failed: " + why);
            return false;
        }
        ++dropped;
        if (h.type == kHop && h.val == 2) {   // a packed hop: its blocks say their length
            for (uint64_t left = h.bytes; left > 0;) {
                uint32_t len = 0, k = 0;
                if (!read_all(g_data, &len, 4) || len < 4 || len > 4 * kBlk) {
                    fail(err, "the connection is lost");
                    return false;
                }
                junk.resize(len);
                if (!read_all(g_data, junk.data(), len)) {
                    fail(err, "the connection is lost");
                    return false;
                }
                std::memcpy(&k, junk.data(), 4);
                left -= std::min<uint64_t>(left, k == 0 ? left : k);
            }
            continue;
        }
        for (uint64_t left = h.bytes; left > 0;) {
            const size_t k = (size_t) std::min<uint64_t>(left, 1 << 20);
            junk.resize(k);
            if (!read_all(g_data, junk.data(), k)) {
                fail(err, "the connection is lost");
                return false;
            }
            left -= k;
        }
    }
}

bool end_front(int64_t pos, int32_t flags, bool& agree, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    Hdr h{kMagic, kEnd, pos, 0, flags, 0}, p{};
    if (!send_msg(h, nullptr, 0)) { fail(err, "END send failed"); return false; }
    int dropped = 0;
    if (!drain_until(kAck, p, dropped, err)) return false;
    agree = p.val != 0 && p.pos == pos;
    if (dropped > 0) std::fprintf(stderr, "glm link: %d stale message(s) from the back dropped at the request's end\n", dropped);
    return true;
}

bool end_back(int64_t pos, int32_t& flags, bool& agree, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    if (!g_cancel) {
        Hdr p{};
        int dropped = 0;
        if (!drain_until(kEnd, p, dropped, err)) return false;
        g_end_pos = p.pos;
        g_end_flags = p.val;
        if (dropped > 0) std::fprintf(stderr, "glm link: %d stale message(s) from the front dropped at the request's end\n", dropped);
    }
    flags = g_end_flags;
    agree = g_end_pos == pos;
    Hdr h{kMagic, kAck, pos, 0, agree ? 1 : 0, 0};
    g_cancel = false;
    if (!send_msg(h, nullptr, 0)) { fail(err, "ACK send failed"); return false; }
    return true;
}

void ctl_send_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(g_ctl_mu);
    if (g_ctl < 0) return;
    std::string l = line;
    l.push_back('\n');
    if (!write_all(g_ctl, l.data(), l.size())) {
        std::string err;
        fail(err, "control line send failed");
    }
}

void ctl_close() {
    std::lock_guard<std::mutex> lk(g_ctl_mu);
    if (g_ctl >= 0) ::shutdown(g_ctl, SHUT_WR);
}

void report() {
    std::fprintf(stderr, "glm link: %s %llu hops (%.1f MB), %llu tokens, %.1f ms sending, %.1f ms receiving\n", role_name(),
                 (unsigned long long) s_hops, (double) s_hop_bytes / 1e6, (unsigned long long) s_toks, s_send_ms,
                 s_recv_ms);
}
#endif

}  // namespace strata::core::glmlink
