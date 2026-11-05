// glm_link: see strata/core/glm_link.hpp
#include "strata/core/glm_link.hpp"

#include <algorithm>
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
#include <sys/random.h>
#include <sys/uio.h>
#include <poll.h>
#include <unistd.h>
#endif
#include <cerrno>
#include <utility>
#include <vector>

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
    if (host.empty()) host = "127.0.0.1";   // (never "every interface" by omission: the back binds loopback)
    return true;
}

// THE BACK'S PEERS: STRATA_GLM_LINK_ALLOW / MAYA_GLM_LINK_ALLOW / MAYA_LINK_ALLOW = comma-separated IPv4 addresses the
// back accepts a front from; unset: loopback (127.0.0.0/8) only.  Any other source is closed at accept().
bool peer_allowed(int fd, std::string& ip) {
    struct sockaddr_in a {};
    socklen_t al = sizeof a;
    ip = "?";
    if (::getpeername(fd, (struct sockaddr*) &a, &al) != 0 || a.sin_family != AF_INET) return false;
    char buf[INET_ADDRSTRLEN] = {0};
    if (::inet_ntop(AF_INET, &a.sin_addr, buf, sizeof buf) == nullptr) return false;
    ip = buf;
    const char* al_env = std::getenv("STRATA_GLM_LINK_ALLOW");
    if (al_env == nullptr || !al_env[0]) al_env = std::getenv("MAYA_GLM_LINK_ALLOW");
    if (al_env == nullptr || !al_env[0]) al_env = std::getenv("MAYA_LINK_ALLOW");
    if (al_env == nullptr || !al_env[0]) return (ntohl(a.sin_addr.s_addr) >> 24) == 127;
    const std::string list = al_env;
    size_t i = 0;
    while (i <= list.size()) {
        size_t j = list.find(',', i);
        if (j == std::string::npos) j = list.size();
        std::string t = list.substr(i, j - i);
        while (!t.empty() && t.front() == ' ') t.erase(t.begin());
        while (!t.empty() && t.back() == ' ') t.pop_back();
        struct in_addr x {};
        if (!t.empty() && ::inet_pton(AF_INET, t.c_str(), &x) == 1 && x.s_addr == a.sin_addr.s_addr) return true;
        i = j + 1;
    }
    return false;
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

// LINK AUTHENTICATION: STRATA_GLM_LINK_KEY_FILE / MAYA_GLM_LINK_KEY_FILE names a file holding a shared secret (at least
// 32 bytes; trailing whitespace is dropped).  With it, each connection (DATA, CTRL) proves both ways that the other
// side holds the same key before anything else crosses: the back sends a 32-byte nonce, the front answers with its
// own nonce and HMAC-SHA256(key, "front" label | tag | back nonce | front nonce), the back checks it (constant time;
// a failure is logged and the connection closed, the back keeps listening) and answers with HMAC-SHA256(key, "back"
// label | tag | front nonce | back nonce), which the front checks.  The env set and the file missing or short: the
// process does not start.  Unset: no authentication (the allowlist alone).  Self-contained SHA-256 (FIPS 180-4).
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char buf[64];
    uint64_t len = 0;
    size_t fill = 0;
    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const void* data, size_t n) {
        const unsigned char* p = (const unsigned char*) data;
        len += n;
        while (n > 0) {
            const size_t k = std::min(n, (size_t) 64 - fill);
            std::memcpy(buf + fill, p, k);
            fill += k; p += k; n -= k;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    void final(unsigned char out[32]) {
        const uint64_t bits = len * 8;
        const unsigned char one = 0x80, zero = 0;
        update(&one, 1);
        while (fill != 56) update(&zero, 1);
        unsigned char lb[8];
        for (int i = 0; i < 8; ++i) lb[i] = (unsigned char) (bits >> (56 - 8 * i));
        update(lb, 8);
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) out[4 * i + j] = (unsigned char) (h[i] >> (24 - 8 * j));
    }
};

// HMAC-SHA256 (RFC 2104) over the concatenation of parts
void hmac_sha256(const std::string& key, const std::vector<std::pair<const void*, size_t>>& parts, unsigned char out[32]) {
    unsigned char k0[64] = {0};
    if (key.size() > 64) {
        Sha256 s;
        s.update(key.data(), key.size());
        s.final(k0);
    } else {
        std::memcpy(k0, key.data(), key.size());
    }
    unsigned char ipad[64], opad[64], inner[32];
    for (int i = 0; i < 64; ++i) { ipad[i] = k0[i] ^ 0x36; opad[i] = k0[i] ^ 0x5c; }
    Sha256 si;
    si.update(ipad, 64);
    for (const auto& p : parts) si.update(p.first, p.second);
    si.final(inner);
    Sha256 so;
    so.update(opad, 64);
    so.update(inner, 32);
    so.final(out);
}

bool ct_equal(const unsigned char* a, const unsigned char* b, size_t n) {
    volatile unsigned char d = 0;
    for (size_t i = 0; i < n; ++i) d = d | (unsigned char) (a[i] ^ b[i]);
    return d == 0;
}

std::string g_key;          // the link's shared key (empty: no authentication)
int g_key_state = -1;       // -1 not read yet, 0 failed, 1 read (or no key file set)
std::string g_key_err;

bool load_key() {
    if (g_key_state >= 0) return g_key_state == 1;
    g_key_state = 0;
    const char* path = env2("STRATA_GLM_LINK_KEY_FILE", "MAYA_GLM_LINK_KEY_FILE");
    if (path == nullptr || !path[0]) {
        g_key_state = 1;
        return true;
    }
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        g_key_err = std::string("glm link: MAYA_GLM_LINK_KEY_FILE ") + path + " cannot be read: " + std::strerror(errno);
        return false;
    }
    std::string k;
    char b[256];
    size_t r;
    while ((r = std::fread(b, 1, sizeof b, f)) > 0 && k.size() < 4096) k.append(b, r);
    std::fclose(f);
    while (!k.empty() && (k.back() == '\n' || k.back() == '\r' || k.back() == ' ' || k.back() == '\t')) k.pop_back();
    if (k.size() < 32) {
        g_key_err = std::string("glm link: MAYA_GLM_LINK_KEY_FILE ") + path + " holds " + std::to_string(k.size()) +
                    " bytes, at least 32 are needed";
        return false;
    }
    g_key = std::move(k);
    g_key_state = 1;
    return true;
}

#if !defined(_WIN32)
constexpr size_t kNonce = 32;
const char kLabelFront[] = "maya-glm-link-auth-v1 front";
const char kLabelBack[] = "maya-glm-link-auth-v1 back";

bool random_bytes(unsigned char* p, size_t n) {
    while (n > 0) {
        const ssize_t r = ::getrandom(p, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r;
        n -= (size_t) r;
    }
    return true;
}

// the back, on an accepted connection (its tag read): true when the front proved the key and got the back's proof
bool auth_back(int fd, const char tag[4], std::string& why) {
    if (g_key.empty()) return true;
    unsigned char nb[kNonce], nf[kNonce], mf[32], want[32], mb[32];
    if (!random_bytes(nb, kNonce)) { why = "no random bytes for the nonce"; return false; }
    if (!write_all(fd, nb, kNonce)) { why = "the nonce did not go out"; return false; }
    if (!read_all(fd, nf, kNonce) || !read_all(fd, mf, 32)) { why = "no authentication reply (no key on the front?)"; return false; }
    hmac_sha256(g_key, {{kLabelFront, sizeof kLabelFront - 1}, {tag, 4}, {nb, kNonce}, {nf, kNonce}}, want);
    if (!ct_equal(want, mf, 32)) { why = "authentication failed (wrong link key)"; return false; }
    hmac_sha256(g_key, {{kLabelBack, sizeof kLabelBack - 1}, {tag, 4}, {nf, kNonce}, {nb, kNonce}}, mb);
    if (!write_all(fd, mb, 32)) { why = "the proof did not go out"; return false; }
    return true;
}

// the front, on a connection (its tag sent): true when both proofs check out
bool auth_front(int fd, const char tag[4], std::string& why) {
    if (g_key.empty()) return true;
    unsigned char nb[kNonce], nf[kNonce], mf[32], want[32], mb[32];
    struct timeval tv {10, 0}, t0v {0, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    bool ok = false;
    if (!read_all(fd, nb, kNonce)) why = "the back sent no nonce (no key on the back, or it refused this address)";
    else if (!random_bytes(nf, kNonce)) why = "no random bytes for the nonce";
    else {
        hmac_sha256(g_key, {{kLabelFront, sizeof kLabelFront - 1}, {tag, 4}, {nb, kNonce}, {nf, kNonce}}, mf);
        if (!write_all(fd, nf, kNonce) || !write_all(fd, mf, 32)) why = "the proof did not go out";
        else if (!read_all(fd, mb, 32)) why = "the back closed the connection: it rejected this side's proof (wrong link key)";
        else {
            hmac_sha256(g_key, {{kLabelBack, sizeof kLabelBack - 1}, {tag, 4}, {nf, kNonce}, {nb, kNonce}}, want);
            ok = ct_equal(want, mb, 32);
            if (!ok) why = "the back's proof is wrong (wrong link key)";
        }
    }
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t0v, sizeof t0v);
    return ok;
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

bool key_ready(std::string& err) {
    if (load_key()) return true;
    err = g_key_err;
    return false;
}

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
        std::string why;
        if (!auth_front(fd, tag, why)) {
            ::close(fd);
            close_fds();
            err = std::string("glm link: link authentication failed on the ") + tag + " connection: " + why;
            std::fprintf(stderr, "%s\n", err.c_str());
            return false;
        }
        (which == 0 ? g_data : g_ctl) = fd;
    }
    return true;
}

// the back: the front's two connections on the listening socket
static bool accept_pair(std::string& err) {
    std::string pair_ip;   // both connections of a pair from one address
    while (g_data < 0 || g_ctl < 0) {
        const int fd = ::accept(g_listen, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) continue;
            err = std::string("glm link: accept: ") + std::strerror(errno);
            return false;
        }
        std::string ip;
        if (!peer_allowed(fd, ip) || (!pair_ip.empty() && ip != pair_ip)) {
            std::fprintf(stderr, "glm link: refused a connection from %s (not in MAYA_GLM_LINK_ALLOW%s)\n", ip.c_str(),
                         pair_ip.empty() ? "" : ", or not the pair's address");
            ::close(fd);
            continue;
        }
        tune(fd);
        char tag[4];
        struct timeval tv {10, 0}, t0v {0, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        if (!read_all(fd, tag, 4)) { ::close(fd); continue; }
        if (std::memcmp(tag, "DATA", 4) != 0 && std::memcmp(tag, "CTRL", 4) != 0) { ::close(fd); continue; }
        std::string why;
        if (!auth_back(fd, tag, why)) {
            std::fprintf(stderr, "glm link: refused a connection from %s: %s\n", ip.c_str(), why.c_str());
            ::close(fd);
            continue;
        }
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t0v, sizeof t0v);
        if (std::memcmp(tag, "DATA", 4) == 0) {
            if (g_data >= 0) ::close(g_data);   // (a front that started over: its newest pair stands)
            g_data = fd;
            pair_ip = ip;
        } else if (std::memcmp(tag, "CTRL", 4) == 0) {
            if (g_ctl >= 0) ::close(g_ctl);
            g_ctl = fd;
            pair_ip = ip;
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
    if (!key_ready(err)) return false;
    std::fprintf(stderr, "glm link: authentication %s\n",
                 g_key.empty() ? "OFF (no MAYA_GLM_LINK_KEY_FILE; the allowlist only)" : "on (HMAC-SHA256 challenge-response, both ways)");
    const auto t0 = std::chrono::steady_clock::now();
    if (role() == kFront) {
        double wait_s = 1800;
        if (const char* w = std::getenv("STRATA_GLM_LINK_WAIT")) wait_s = std::atof(w);
        std::fprintf(stderr, "glm link: front connecting to %s:%s\n", host.c_str(), port.c_str());
        if (!connect_pair(host, port, wait_s, err)) return false;
    } else {
        // the address to listen on: STRATA_GLM_LINK_BIND / MAYA_GLM_LINK_BIND, else the --link host (empty: loopback)
        if (const char* b = env2("STRATA_GLM_LINK_BIND", "MAYA_GLM_LINK_BIND"); b != nullptr && b[0]) host = b;
        struct addrinfo hints {}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_NUMERICSERV;
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
        {
            const char* al = std::getenv("STRATA_GLM_LINK_ALLOW");
            if (al == nullptr || !al[0]) al = std::getenv("MAYA_GLM_LINK_ALLOW");
            if (al == nullptr || !al[0]) al = std::getenv("MAYA_LINK_ALLOW");
            std::fprintf(stderr, "glm link: back listening on %s:%s, fronts accepted from %s\n", host.c_str(), port.c_str(),
                         al != nullptr && al[0] ? al : "loopback only (set MAYA_GLM_LINK_ALLOW)");
        }
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
// p: the whole block as read (buf bytes: its u32 length + 4); false on a block that does not fit its own length
static bool unpack_block(const uint8_t* p, size_t buf, size_t k_want, uint32_t* dst) {
    PackHdr h;
    if (buf < sizeof h) return false;
    std::memcpy(&h, p, sizeof h);
    if (h.k != k_want || h.n_esc > h.k || (size_t) h.len + 4 != buf) return false;
    if (sizeof(PackHdr) + (size_t) h.k + ((size_t) h.k + 1) / 2 + (size_t) h.n_esc > buf) return false;
    const uint8_t* lo = p + sizeof(PackHdr);
    const uint8_t* nib = lo + h.k;
    const uint8_t* esc = nib + (h.k + 1) / 2;
    size_t e = 0;
    for (size_t j = 0; j < h.k; ++j) {
        const uint8_t c = (uint8_t) ((nib[j >> 1] >> ((j & 1) * 4)) & 15);
        if (c == 15 && (e >= h.n_esc || esc + e >= p + buf)) return false;   // (every escape inside the block)
        const uint8_t hi = c == 15 ? esc[e++] : h.table[c];
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
// an ERR's text: at most kMaxErrBytes (a longer one is not read - the caller drops the link)
constexpr uint64_t kMaxErrBytes = 4096;
constexpr uint64_t kMaxHopBytes = 4ull << 30;   // a HOP's payload bound for dropped messages (an 8K chunk is 512 MB)
static bool read_err(const Hdr& h, std::string& why) {
    if (h.bytes > kMaxErrBytes) return false;
    why.assign((size_t) h.bytes, '\0');
    if (h.bytes > 0 && !read_all(g_data, &why[0], (size_t) h.bytes)) why = "?";
    return true;
}

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
        std::string why;
        if (!read_err(h, why)) {
            fail(err, "an ERR of " + std::to_string((unsigned long long) h.bytes) + " bytes (over 4 KiB: a protocol error)");
            return false;
        }
        fail(err, "the other side failed: " + why);
        return false;
    }
    if (h.type != kHop && h.bytes != 0) {   // (only a HOP and an ERR carry a payload)
        fail(err, "message " + std::to_string(h.type) + " with a payload of " +
                      std::to_string((unsigned long long) h.bytes) + " bytes (a protocol error)");
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
            if (!(ok = read_all(g_data, st.data() + 4, len))) break;
            if (!unpack_block(st.data(), (size_t) len + 4, k, out + i)) {
                fail(err, "a packed HOP block that does not fit its length (a protocol error)");
                return false;
            }
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
    const size_t wn = std::min<size_t>(why.size(), 4096);   // (the receiver drops a longer ERR as a protocol error)
    Hdr e{kMagic, kErr, 0, 0, 0, (uint64_t) wn};
    send_msg(e, why.data(), wn);
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
            std::string why;
            if (!read_err(h, why)) {
                fail(err, "an ERR of " + std::to_string((unsigned long long) h.bytes) + " bytes (over 4 KiB: a protocol error)");
                return false;
            }
            fail(err, "the other side failed: " + why);
            return false;
        }
        // (a dropped message's payload: a HOP's rows at most - the context's whole prompt chunk, fp32)
        if ((h.type != kHop && h.bytes != 0) || h.bytes > kMaxHopBytes) {
            fail(err, "message " + std::to_string(h.type) + " with a payload of " +
                          std::to_string((unsigned long long) h.bytes) + " bytes (a protocol error)");
            return false;
        }
        ++dropped;
        if (h.type == kHop && h.val == 2) {   // a packed hop: its blocks say their length
            for (uint64_t left = h.bytes; left > 0;) {
                uint32_t len = 0, k = 0;
                if (!read_all(g_data, &len, 4) || len < sizeof(PackHdr) - 4 || len > 4 * kBlk) {
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
