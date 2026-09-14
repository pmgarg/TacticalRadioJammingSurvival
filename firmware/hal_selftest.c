/* hal_selftest.c -- prove the Mode H port behaves like the world it stands in for.
 *
 * Same discipline as tests/test_contract_effects.py on the simulation side: every action
 * must produce a measurable change, and the two situations that must never be confused --
 * an attacker and a fade -- must separate on the features the design says they separate on.
 *
 *   cc -std=c11 -Wall -Wextra -o haltest hal_host.c hal_selftest.c && ./haltest
 */
#include "hal.h"

#include <stdio.h>
#include <string.h>

void hal_host_scenario(int emitter, uint8_t emitter_ch, float emitter_dbm,
                       bool fading, int dead_peer);

static int pass_n, fail_n;

static void check(const char *name, int ok, const char *detail)
{
    if (ok) { pass_n++; } else { fail_n++; }
    printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
           detail && detail[0] ? "  -- " : "", detail ? detail : "");
}

/* mean over `n` samples, so a single noisy sample cannot decide a test */
static void sample(int n, float *floor_out, float *pdr_out, float *s4_out)
{
    hal_obs_t o;
    float f = 0, p = 0;
    unsigned se = 0, sm = 0, ie = 0, im = 0;
    for (int k = 0; k < n; ++k)
    {
        hal_sense(&o);
        f += o.noise_floor_dbm;
        unsigned ok = 0, ex = 0;
        for (int i = 0; i < o.n_peers; ++i) { ok += o.rx_ok[i]; ex += o.rx_expected[i]; }
        p += ex ? (float)ok / (float)ex : 0.0f;
        se += o.shadow_expected; sm += o.shadow_missed;
        ie += o.silent_expected; im += o.silent_missed;
    }
    *floor_out = f / (float)n;
    *pdr_out = p / (float)n;
    *s4_out = (se && ie) ? ((float)sm / (float)se - (float)im / (float)ie) : 0.0f;
}

int main(void)
{
    float f, p, s4, f2, p2, s42;
    char buf[160];

    printf("======================================================================\n");
    printf("MODE H PORT SELF-TEST  (hal_host.c -- no hardware required)\n");
    printf("======================================================================\n");

    printf("\n=== 1. A quiet world is quiet ===\n");
    hal_init(0, 6);
    hal_host_scenario(0, 0, 0, false, -1);
    sample(40, &f, &p, &s4);
    snprintf(buf, sizeof buf, "floor=%.1f dBm  pdr=%.2f", f, p);
    check("no emitter: floor near nominal and delivery healthy", f < -90.0f && p > 0.8f, buf);

    printf("\n=== 2. S2 -- an emitter lifts the floor, a fade does NOT ===\n");
    hal_init(0, 6);
    hal_host_scenario(2, 6, -75.0f, false, -1);          /* spot jammer on our channel */
    sample(40, &f, &p, &s4);
    hal_init(0, 6);
    hal_host_scenario(0, 0, 0, true, -1);                /* fading, no attacker        */
    sample(40, &f2, &p2, &s42);
    snprintf(buf, sizeof buf, "jammed floor=%.1f  faded floor=%.1f  (faded pdr=%.2f)", f, f2, p2);
    check("jamming raises the floor >10 dB above fading", (f - f2) > 10.0f, buf);
    snprintf(buf, sizeof buf, "pdr=%.2f with the floor still at %.1f dBm", p2, f2);
    check("fading degrades delivery WITHOUT raising the floor -- the FP trap",
          p2 < 0.9f && f2 < -90.0f, buf);

    printf("\n=== 3. S5 -- the scan finds the hot channel, and only it ===\n");
    hal_init(0, 6);
    hal_host_scenario(2, 3, -70.0f, false, -1);
    hal_scan_t sc;
    hal_scan(&sc, 20);
    int hot = 0;
    for (int c = 1; c < HAL_N_CHANNELS; ++c)
    {
        if (sc.noise_dbm[c] > sc.noise_dbm[hot]) { hot = c; }
    }
    int quiet = 0;
    for (int c = 0; c < HAL_N_CHANNELS; ++c)
    {
        if (sc.noise_dbm[c] < -90.0f) { quiet++; }
    }
    snprintf(buf, sizeof buf, "hottest=ch%d (configured 3), %d/%d channels still quiet",
             hot + 1, quiet, HAL_N_CHANNELS);
    check("scan identifies the jammed channel and leaves the rest clean",
          hot + 1 == 3 && quiet >= 4, buf);

    printf("\n=== 4. S4' -- reactive is visible in TIMING, not in the floor ===\n");
    hal_init(0, 6);
    hal_host_scenario(3, 6, -70.0f, false, -1);
    sample(40, &f, &p, &s4);
    snprintf(buf, sizeof buf, "S4'=%+.2f with the floor at %.1f dBm", s4, f);
    check("reactive: S4' strongly positive while the floor stays quiet",
          s4 > 0.3f && f < -90.0f, buf);
    hal_init(0, 6);
    hal_host_scenario(1, 0, -76.0f, false, -1);
    sample(40, &f2, &p2, &s42);
    snprintf(buf, sizeof buf, "reactive S4'=%+.2f  barrage S4'=%+.2f", s4, s42);
    check("S4' separates reactive from barrage", s4 > s42, buf);

    printf("\n=== 5. Every act() actually changes something ===\n");
    hal_init(0, 6);
    hal_host_scenario(2, 6, -70.0f, false, -1);
    sample(20, &f, &p, &s4);
    check("hop_channel moves us off the jammed channel", hal_set_channel(2) == 0, "");
    sample(20, &f2, &p2, &s42);
    snprintf(buf, sizeof buf, "floor %.1f -> %.1f dBm after hopping 6 -> 2", f, f2);
    check("hop_channel: measured floor drops", f2 < f - 10.0f, buf);

    hal_init(0, 6);
    hal_host_scenario(0, 0, 0, true, -1);
    hal_obs_t o;
    hal_sense(&o);
    float rev_lo = o.rev_pdr[0];
    hal_set_tx_power(20);
    hal_sense(&o);
    snprintf(buf, sizeof buf, "reverse pdr %.2f -> %.2f", rev_lo, o.rev_pdr[0]);
    check("set_tx_power moves the REVERSE link (the direction it can move)",
          o.rev_pdr[0] >= rev_lo, buf);

    hal_set_tx(false);
    hal_sense(&o);
    check("silent_listen stops our transmissions", o.tx_attempts == 0, "");
    hal_set_tx(true);

    printf("\n=== 6. node_loss: one peer silent, channel clean ===\n");
    hal_init(0, 6);
    hal_host_scenario(0, 0, 0, false, 1);
    hal_sense(&o);
    check("the dead peer is silent", o.rx_ok[1] == 0 && o.hb_gap_ms[1] > 1000u, "");
    check("the other peers are fine", o.hb_gap_ms[0] < 1000u, "");
    check("neighbor_probe answers for a live peer", hal_neighbor_probe(0, 50) == 1, "");
    check("neighbor_probe reports the dead one", hal_neighbor_probe(1, 50) == 0, "");
    sample(20, &f, &p, &s4);
    snprintf(buf, sizeof buf, "floor=%.1f dBm", f);
    check("node_loss does NOT look like jamming (floor stays quiet)", f < -90.0f, buf);

    printf("\n======================================================================\n");
    printf("MODE H PORT: %d passed, %d failed\n", pass_n, fail_n);
    printf("======================================================================\n");
    return fail_n ? 1 : 0;
}
