/* hal_host.c -- deterministic desktop implementation of hal.h.
 *
 * Not a simulator and not a toy: its job is to let the ENTIRE on-device stack -- percept,
 * envelope, rules, net -- be compiled and exercised on a laptop before any hardware is
 * bought. A port that has never been run is a port that does not work, and finding that
 * out with a soldering iron in your hand is the expensive way.
 *
 * It models the four things the agent actually keys on, and nothing else:
 *   an emitter raises the floor on the channels it covers,
 *   a reactive emitter only fires after OUR transmissions,
 *   fading degrades delivery WITHOUT raising the floor  <- the false-positive trap,
 *   a dead peer goes silent while the channel stays clean.
 *
 *   cc -std=c11 -Wall -Wextra -o haltest hal_host.c hal_selftest.c && ./haltest
 */
#include "hal.h"

#include <stdlib.h>
#include <string.h>

#define NOMINAL_NOISE_DBM (-96.0f)

static struct
{
    uint8_t  me, channel, slot, n_slots;
    int8_t   txp;
    bool     tx_on;
    uint64_t t_us;
    uint32_t seed;
    /* scripted world, set by hal_host_scenario() */
    int      emitter;                    /* 0 none, 1 barrage, 2 spot, 3 reactive       */
    uint8_t  emitter_ch;
    float    emitter_dbm;
    bool     fading;
    int      dead_peer;                  /* -1 none                                     */
    uint8_t  n_peers;
} S;

static float frnd(void)
{
    S.seed = S.seed * 1103515245u + 12345u;
    return (float)((S.seed >> 16) & 0x7fff) / 32767.0f;
}

void hal_host_scenario(int emitter, uint8_t emitter_ch, float emitter_dbm,
                       bool fading, int dead_peer)
{
    S.emitter = emitter;
    S.emitter_ch = emitter_ch;
    S.emitter_dbm = emitter_dbm;
    S.fading = fading;
    S.dead_peer = dead_peer;
}

int hal_init(uint8_t my_index, uint8_t channel)
{
    memset(&S, 0, sizeof S);
    S.me = my_index;
    S.channel = channel;
    S.txp = 16;
    S.tx_on = true;
    S.seed = 12345u + my_index;
    S.dead_peer = -1;
    S.n_peers = 3;
    return 0;
}

bool hal_have_noise(void) { return true; }

/* energy this node sees on `ch`, in dBm */
static float floor_on(uint8_t ch)
{
    float f = NOMINAL_NOISE_DBM + frnd() * 0.5f;
    if (S.emitter == 1)                                  /* barrage: whole band        */
    {
        f = S.emitter_dbm;
    }
    else if (S.emitter == 2)                             /* spot: one channel + mask   */
    {
        int d = (int)ch - (int)S.emitter_ch;
        if (d < 0) { d = -d; }
        if (d == 0)      { f = S.emitter_dbm; }
        else if (d == 1) { f = S.emitter_dbm - 8.0f; }
        else if (d == 2) { f = S.emitter_dbm - 28.0f; }
    }
    /* emitter == 3 (reactive) deliberately does NOT lift the floor: it only radiates
     * just after our transmissions, so a min-hold estimator cannot see it. That is the
     * whole reason S4' exists and S2 does not carry this family. */
    return f;
}

void hal_sense(hal_obs_t *o)
{
    memset(o, 0, sizeof *o);
    S.t_us += 100000;                                     /* one 100 ms sample period  */
    o->t_us = S.t_us;
    o->channel = S.channel;
    o->n_peers = S.n_peers;
    o->noise_floor_dbm = floor_on(S.channel);
    o->own_tx_active = S.tx_on;

    float hot = o->noise_floor_dbm - NOMINAL_NOISE_DBM;    /* dB above quiet            */
    for (uint8_t i = 0; i < S.n_peers; ++i)
    {
        bool dead = ((int)i == S.dead_peer);
        float base = -62.0f - 3.0f * (float)i;
        float loss = 0.0f;
        if (hot > 3.0f)  { loss = (hot > 20.0f) ? 0.95f : 0.45f; }
        if (S.fading)    { loss += 0.5f * frnd(); base -= 18.0f + 8.0f * frnd(); }
        if (!S.tx_on)    { loss *= 1.0f; }
        if (loss > 0.98f) { loss = 0.98f; }

        o->rssi_dbm[i] = dead ? -110.0f : base;
        o->rx_expected[i] = 1;
        o->rx_ok[i] = dead ? 0u : (frnd() > loss ? 1u : 0u);
        o->hb_gap_ms[i] = dead ? 9000u : (o->rx_ok[i] ? 0u : 100u);
        o->rev_rssi_dbm[i] = dead ? -110.0f : base + (float)(S.txp - 16);
        o->rev_pdr[i] = dead ? 0.0f : (1.0f - loss) * (S.txp >= 16 ? 1.0f : 0.75f);
        o->rev_age_ms[i] = dead ? 9000u : 100u;
    }

    o->tx_attempts = S.tx_on ? 10u : 0u;
    o->tx_acked = (uint32_t)((float)o->tx_attempts * (hot > 3.0f ? 0.3f : 0.95f));
    o->tx_defer_us = (uint32_t)(hot > 3.0f ? 4000 : 200);

    /* S4': a reactive emitter concentrates its damage in the window after OUR TX */
    if (S.tx_on)
    {
        o->shadow_expected = 4;
        o->silent_expected = 6;
        o->shadow_missed = (S.emitter == 3) ? 3u : (hot > 3.0f ? 2u : 0u);
        o->silent_missed = (S.emitter == 3) ? 0u : (hot > 3.0f ? 3u : 0u);
    }
}

int hal_scan(hal_scan_t *o, uint32_t dwell_ms)
{
    memset(o, 0, sizeof *o);
    o->t_us = S.t_us;
    for (uint8_t c = 0; c < HAL_N_CHANNELS; ++c)
    {
        o->noise_dbm[c] = floor_on((uint8_t)(c + 1));
    }
    o->valid = 0xFF;
    S.t_us += (uint64_t)dwell_ms * 1000u * HAL_N_CHANNELS;   /* a scan costs deafness  */
    return 0;
}

int hal_neighbor_probe(uint8_t peer, uint32_t timeout_ms)
{
    (void)timeout_ms;
    if (peer >= S.n_peers) { return -1; }
    return ((int)peer == S.dead_peer) ? 0 : 1;
}

int hal_set_channel(uint8_t ch)
{
    if (ch < 1 || ch > HAL_N_CHANNELS) { return -1; }
    S.channel = ch;
    return 0;
}

int hal_set_tx_power(int8_t dbm) { S.txp = dbm; return 0; }

int hal_set_slot(uint8_t slot, uint8_t n_slots)
{
    S.slot = slot;
    S.n_slots = n_slots;
    return 0;
}

int hal_set_tx(bool enabled) { S.tx_on = enabled; return 0; }

int hal_declare_lost(void) { return 0; }
