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
#include <unistd.h>
#include <cerrno>
#endif

namespace strata::core::glmlink {
namespace {

int g_data = -1, g_ctl = -1;
std::atomic<bool> g_up{false};
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
void ctl_send_line(const std::string&) {}
void ctl_close() {}
void report() {}
#else
bool open(int split_at, int64_t max_ctx, int my_chunk, int& peer_chunk, std::string& err) {
    const char* lk = env2("STRATA_GLM_LINK", "MAYA_GLM_LINK");
    std::string host, port;
    if (lk == nullptr || !split_hostport(lk, host, port)) {
        err = "glm link: STRATA_GLM_LINK=host:port (or --link) is needed with --role";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (role() == kFront) {
        double wait_s = 1800;
        if (const char* w = std::getenv("STRATA_GLM_LINK_WAIT")) wait_s = std::atof(w);
        std::fprintf(stderr, "glm link: front connecting to %s:%s\n", host.c_str(), port.c_str());
        for (int which = 0; which < 2; ++which) {
            int fd = -1;
            while ((fd = connect_once(host, port)) < 0) {
                if (ms_since(t0) > wait_s * 1000.0) {
                    err = "glm link: no back at " + host + ":" + port + " after " + std::to_string((int) wait_s) + " s";
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            const char* tag = which == 0 ? "DATA" : "CTRL";
            if (!write_all(fd, tag, 4)) {
                ::close(fd);
                err = "glm link: the connection's tag did not go out";
                return false;
            }
            (which == 0 ? g_data : g_ctl) = fd;
        }
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
            err = std::string("glm link: cannot listen on ") + lk + ": " + std::strerror(errno);
            return false;
        }
        freeaddrinfo(res);
        std::fprintf(stderr, "glm link: back listening on %s:%s\n", host.c_str(), port.c_str());
        while (g_data < 0 || g_ctl < 0) {
            const int fd = ::accept(ls, nullptr, nullptr);
            if (fd < 0) {
                if (errno == EINTR) continue;
                ::close(ls);
                err = std::string("glm link: accept: ") + std::strerror(errno);
                return false;
            }
            tune(fd);
            char tag[4];
            if (!read_all(fd, tag, 4)) { ::close(fd); continue; }
            if (std::memcmp(tag, "DATA", 4) == 0 && g_data < 0) g_data = fd;
            else if (std::memcmp(tag, "CTRL", 4) == 0 && g_ctl < 0) g_ctl = fd;
            else ::close(fd);
        }
        ::close(ls);
    }
    Hdr h{kMagic, kHello, kVersion, split_at, my_chunk, (uint64_t) max_ctx}, p{};
    if (!send_msg(h, nullptr, 0) || !read_all(g_data, &p, sizeof p)) {
        err = "glm link: HELLO failed";
        return false;
    }
    if (p.magic != kMagic || p.type != kHello || p.pos != kVersion) {
        err = "glm link: the other side speaks another protocol";
        return false;
    }
    if (p.n != split_at || (int64_t) p.bytes != max_ctx) {
        err = "glm link: the other side splits at " + std::to_string(p.n) + " with context " +
              std::to_string((long long) p.bytes) + ", this one at " + std::to_string(split_at) + " with " +
              std::to_string((long long) max_ctx);
        return false;
    }
    peer_chunk = p.val;
    g_up.store(true);
    std::fprintf(stderr, "glm link: %s up in %.1f s (boundary layer %d, context %lld, prompt chunks %d / %d)\n",
                 role_name(), ms_since(t0) / 1000.0, split_at, (long long) max_ctx, my_chunk, peer_chunk);
    std::atexit(report);
    return true;
}

bool send_hop(int64_t pos, int n, const void* data, size_t bytes, std::string& err) {
    if (!up()) { err = "glm link: down"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    Hdr h{kMagic, kHop, pos, n, 0, (uint64_t) bytes};
    if (!send_msg(h, data, bytes)) {
        fail(err, "HOP send failed");
        return false;
    }
    s_send_ms += ms_since(t0);
    ++s_hops;
    s_hop_bytes += bytes;
    return true;
}

// read one header; an ERR from the other side becomes this side's error
static bool recv_hdr(Hdr& h, uint32_t want, std::string& err) {
    if (!read_all(g_data, &h, sizeof h) || h.magic != kMagic) {
        fail(err, "the connection is lost");
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
    if (h.pos != pos || h.n != n || h.bytes != bytes) {
        fail(err, "HOP for positions " + std::to_string((long long) h.pos) + "+" + std::to_string(h.n) +
                      ", expected " + std::to_string((long long) pos) + "+" + std::to_string(n) + " (out of step)");
        return false;
    }
    if (!read_all(g_data, dst, bytes)) {
        fail(err, "the connection is lost in a HOP");
        return false;
    }
    s_recv_ms += ms_since(t0);
    ++s_hops;
    s_hop_bytes += bytes;
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
