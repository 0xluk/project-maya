// glm_link: one GLM layer split across TWO PROCESSES (two boxes) over TCP - Phase 1 of the two-box split.
//
// Off unless STRATA_GLM_ROLE (or MAYA_GLM_ROLE, or `--role`) is front|back: then nothing here runs and the engine is the
// unmodified one.  With a role the process loads ONE part of the two-part split at the boundary given by --layer-split K
// (front: layers [0, K) + the embedding; back: [K, n_layers) + the NextN block + the output head + the sampler) and
// both processes run the same driver in lock step (SPMD): the front tees every control line it reads (GEN, QUIT, ...)
// to the back, which reads them instead of its stdin, so both make the same model calls in the same order.  Where the
// in-process split copies the residual through a pinned hop buffer, the front sends it over the link and the back
// receives it; the back returns every token it picks.
//
// Two TCP connections, both opened by the front (STRATA_GLM_LINK=host:port; the back listens on it):
//   data: framed messages (Hdr + payload), TCP_NODELAY.  front -> back: HELLO, HOP (residual rows of n positions from
//         pos, n x hc x n_embd fp32); back -> front: HELLO, TOK (a token picked at pos), ERR (text)
//   ctl:  newline-terminated control lines, front -> back (the back's stdin)
// Any link failure is fatal for the request on both sides, and the link stays down (the engine has to be restarted).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core::glmlink {

enum Role : int { kNone = 0, kFront = 1, kBack = 2 };
enum Type : uint32_t { kHello = 1, kHop = 2, kTok = 3, kSample = 4, kErr = 5 };

struct Hdr {
    uint32_t magic;   // kMagic
    uint32_t type;    // Type
    int64_t pos;      // HOP: first position; TOK/SAMPLE: the position the token was picked at; HELLO: protocol version
    int32_t n;        // HOP: positions; HELLO: the split's boundary layer
    int32_t val;      // TOK/SAMPLE: the token id (-1: failed, an ERR follows); HELLO: the prompt chunk (0: none)
    uint64_t bytes;   // payload bytes that follow; HELLO: max context
};
constexpr uint32_t kMagic = 0x4b4e4c4d;   // "MLNK"
constexpr int64_t kVersion = 1;

Role role();                          // from STRATA_GLM_ROLE / MAYA_GLM_ROLE (cached)
const char* role_name();
bool up();                            // the link is connected and healthy

// open the link (front: connect, retrying up to STRATA_GLM_LINK_WAIT seconds, default 1800; back: listen, accept both
// connections) and exchange HELLO.  my_chunk: this side's prompt chunk; peer_chunk: the other side's.
bool open(int split_at, int64_t max_ctx, int my_chunk, int& peer_chunk, std::string& err);

// data channel
bool send_hop(int64_t pos, int n, const void* data, size_t bytes, std::string& err);
// receive a HOP into dst (cap bytes), checking that it carries exactly (pos, n) and `bytes` bytes
bool recv_hop(int64_t pos, int n, void* dst, size_t bytes, std::string& err);
bool send_tok(Type t, int64_t pos, int32_t tok);
// the back failed a request: tell the front why (an ERR in place of what it waits for) and take the link down
void abort_link(const std::string& why);
bool recv_tok(Type t, int64_t pos, int32_t& tok, std::string& err);

// control channel: the front sends each line it reads; the back reads them as its stdin (ctl_fd: -1 off)
void ctl_send_line(const std::string& line);
void ctl_close();                     // front: no more lines (EOF on the back)
int ctl_fd();                         // back: the fd to read the control lines from

// stats (stderr at exit): hops, bytes, time blocked in send/recv
void report();

}  // namespace strata::core::glmlink
