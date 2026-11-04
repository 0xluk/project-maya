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
// Any link failure fails the request on both sides and takes the link down; the front connects again at its next
// command (STRATA_GLM_RELINK_WAIT seconds, default 5) and the back, which stays up, accepts it (Phase 2).
// Every request is bracketed: kBegin (both sides' reuse length and decode mode: a mismatch reads the prompt from 0)
// and kEnd/kAck (the front's end: the back stops wherever it waits - kEnd in place of any message is a cancel - and
// acknowledges with its position; anything stale before the marker is dropped).
// STRATA_GLM_LINK_BF16=1 (default): the boundary rows cross as bf16 (rounded to nearest even, upcast on the back).
// The back listens on STRATA_GLM_LINK_BIND (else the --link host; an empty host is loopback, never every interface) and
// accepts a front only from the addresses in STRATA_GLM_LINK_ALLOW / MAYA_GLM_LINK_ALLOW (comma-separated IPv4;
// unset: loopback only).  Malformed lengths (an ERR over 4 KiB, a payload on a message without one, a packed block
// that does not fit its own length) are protocol errors: the link drops.
// Several local GPUs per side: --layer-split K:b1,b2,.. (glm_model.cu load_pack_place).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core::glmlink {

enum Role : int { kNone = 0, kFront = 1, kBack = 2 };
enum Type : uint32_t { kHello = 1, kHop = 2, kTok = 3, kSample = 4, kErr = 5, kSpec = 6, kCtl = 7, kBegin = 8, kEnd = 9, kAck = 10 };
// kSpec (back -> front, the pipelined spec decode): pos = the tail's position, val = its token y, n = the draft d for
// the position after (-1: none).  kCtl (front -> back): pos = the position the step is for, val = 1 go on / 0 the end

struct Hdr {
    uint32_t magic;   // kMagic
    uint32_t type;    // Type
    int64_t pos;      // HOP: first position; TOK/SAMPLE: the position the token was picked at; HELLO: protocol version
    int32_t n;        // HOP: positions; HELLO: the split's boundary layer
    int32_t val;      // TOK/SAMPLE: the token id (-1: failed, an ERR follows); HELLO: the prompt chunk (0: none)
    uint64_t bytes;   // payload bytes that follow; HELLO: max context
};
constexpr uint32_t kMagic = 0x4b4e4c4d;   // "MLNK"
constexpr int64_t kVersion = 2;

Role role();                          // from STRATA_GLM_ROLE / MAYA_GLM_ROLE (cached)
const char* role_name();
bool up();                            // the link is connected and healthy

// open the link (front: connect, retrying up to STRATA_GLM_LINK_WAIT seconds, default 1800; back: listen, accept both
// connections) and exchange HELLO.  my_chunk: this side's prompt chunk; peer_chunk: the other side's.
bool open(int split_at, int64_t max_ctx, int my_chunk, int& peer_chunk, std::string& err);
// the boundary layer K (set at load, before open: the parts size their RAM tiers over this side's layers only)
void set_split(int split_at);
int split_at();

// data channel
bool send_hop(int64_t pos, int n, const void* data, size_t bytes, std::string& err);
// receive a HOP into dst (cap bytes), checking that it carries exactly (pos, n) and `bytes` bytes
bool recv_hop(int64_t pos, int n, void* dst, size_t bytes, std::string& err);
bool send_tok(Type t, int64_t pos, int32_t tok);
// the back failed a request: tell the front why (an ERR in place of what it waits for) and take the link down
void abort_link(const std::string& why);
bool recv_tok(Type t, int64_t pos, int32_t& tok, std::string& err);
bool send_pair(Type t, int64_t pos, int32_t val, int32_t n);                          // kSpec / kCtl
bool recv_pair(Type t, int64_t pos, int32_t& val, int32_t& n, std::string& err);

// control channel: the front sends each line it reads; the back reads them as its stdin (ctl_fd: -1 off)
void ctl_send_line(const std::string& line);
void ctl_close();                     // front: no more lines (EOF on the back)
int ctl_fd();                         // back: the fd to read the control lines from

// Phase 2: requests, cancel, reconnect
void request_begin();                 // both sides, a request's start: no cancel seen yet
bool cancelled();                     // the back: the front's kEnd came in place of what it waited for (err "cancelled")
// both: the request's start - the front sends (reuse, mode), the back answers with its own
bool begin_exchange(int64_t reuse, int32_t mode, int64_t& peer_reuse, int32_t& peer_mode, std::string& err);
// the front: kEnd{pos, flags}, then everything up to the back's kAck is dropped; agree: the back stands at pos too
bool end_front(int64_t pos, int32_t flags, bool& agree, std::string& err);
// the back: (unless the kEnd came already) everything up to the front's kEnd is dropped, then kAck{pos, agree}
bool end_back(int64_t pos, int32_t& flags, bool& agree, std::string& err);
bool bf16();                          // the hop's wire format
bool idle_check();                    // the front between commands: false if the back has gone (the link is down)
bool reopen(std::string& err);        // the front: connect again (STRATA_GLM_RELINK_WAIT s)
void wake();                          // the back: the old link's data socket shut (a main thread blocked on it returns)
bool reaccept(std::string& err);      // the back: accept the front's next connection and HELLO

// stats (stderr at exit): hops, bytes, time blocked in send/recv
void report();

}  // namespace strata::core::glmlink
