/* hal.h -- the fourth implementation of the RawObs port.
 *
 * The simulator and the drone share everything downstream of RawObs: the 56-feature
 * percept layer, the safety envelope, the induced rules and the distilled net. What they do
 * NOT share is where the numbers come from. In simulation that is ns-3 or refsim; on
 * hardware it is this file.
 *
 * That is the whole sim-to-real gap: six functions. The port is drawn at the TELEMETRY
 * level, not the PHY level, which is why the radio underneath can be ns-3, ESP-NOW, or
 * anything else that can report RSSI, a noise estimate and per-peer delivery.
 *
 * Two implementations ship here:
 *   hal_host.c   -- a deterministic desktop stub. Lets the whole on-device stack be built
 *                   and tested with no hardware at all, which is the difference between
 *                   "the port compiles" and "the port works".
 *   hal_esp32.c  -- ESP-NOW + nRF24L01+ RPD on real 2.4 GHz.
 *
 * Units are fixed and match percept/norm.py EXACTLY. Two feature implementations that
 * disagree on units is the classic sim-to-real trap; there is one set of constants and
 * both sides use it.
 */
#ifndef JS_HAL_H
#define JS_HAL_H

#include <stdbool.h>
#include <stdint.h>

#define HAL_MAX_PEERS   8
#define HAL_N_CHANNELS  8

#ifdef __cplusplus
extern "C" {
#endif

/* Everything the agent may see in one sample period. No ground truth, by construction. */
typedef struct
{
    uint64_t t_us;                       /* monotonic; esp_timer_get_time()             */
    uint8_t  channel;                    /* 1..HAL_N_CHANNELS, the channel we are on    */
    uint8_t  n_peers;                    /* how many entries below are valid            */

    /* --- per peer, forward direction: what WE hear --- */
    float    rssi_dbm[HAL_MAX_PEERS];    /* last frame's RSSI, dBm                      */
    uint32_t rx_ok[HAL_MAX_PEERS];       /* beacons received since the last call        */
    uint32_t rx_expected[HAL_MAX_PEERS]; /* beacons DUE since the last call             */
    uint32_t hb_gap_ms[HAL_MAX_PEERS];   /* silence from this peer                      */

    /* --- per peer, reverse direction: what the PEER says about hearing US ---
     * This costs a handful of bytes per beacon and it is the only way to tell "my
     * transmitter is broken" from "my receiver is broken" (TELEMETRY.md section 2). */
    float    rev_rssi_dbm[HAL_MAX_PEERS];
    float    rev_pdr[HAL_MAX_PEERS];
    uint32_t rev_age_ms[HAL_MAX_PEERS];

    /* --- our own transmit side --- */
    uint32_t tx_attempts;
    uint32_t tx_acked;
    uint32_t tx_defer_us;                /* time spent waiting for a clear channel      */

    /* --- TX-shadow accumulators: the reactive-jamming discriminator (S4') ---
     * Beacons are periodic, so we know when each SHOULD have arrived; we also know
     * exactly when we transmitted. Bucket the misses by whether our own TX was within
     * ~3 ms. Uses ONLY this node's clock -- never compares clocks across nodes, which
     * is what makes it survive on hardware with no time sync. */
    uint32_t shadow_expected, shadow_missed;
    uint32_t silent_expected, silent_missed;

    /* --- energy --- */
    float    noise_floor_dbm;            /* may be unavailable; see hal_have_noise()    */
    bool     own_tx_active;              /* true if we transmitted within the window    */
} hal_obs_t;

/* Per-channel energy, filled by hal_scan(). A scan COSTS: the radio is deaf for dwell_ms. */
typedef struct
{
    float    noise_dbm[HAL_N_CHANNELS];  /* higher = hotter                             */
    uint8_t  valid;                      /* bitmask of channels actually measured       */
    uint64_t t_us;
} hal_scan_t;

/* ---- lifecycle ------------------------------------------------------------------- */
int  hal_init(uint8_t my_index, uint8_t channel);

/* ---- sense: free, continuous. Counters reset on read. ---------------------------- */
void hal_sense(hal_obs_t *out);

/* True when the platform gives a trustworthy noise floor. On some ESP-IDF builds
 * rx_ctrl.noise_floor reads 0 -- in which case S2 comes from the nRF24 scan instead and
 * the agent must know the difference rather than believing a zero. */
bool hal_have_noise(void);

/* ---- diagnose: buys information, costs airtime ----------------------------------- */
int  hal_scan(hal_scan_t *out, uint32_t dwell_ms);
int  hal_neighbor_probe(uint8_t peer, uint32_t timeout_ms);   /* 1 alive, 0 silent, <0 err */

/* ---- act ------------------------------------------------------------------------- */
int  hal_set_channel(uint8_t ch);            /* ~2 ms on ESP32                          */
int  hal_set_tx_power(int8_t dbm);
int  hal_set_slot(uint8_t slot, uint8_t n_slots);  /* TDMA gate; 0 slots = CSMA only    */
int  hal_set_tx(bool enabled);               /* silent_listen                           */
int  hal_declare_lost(void);                 /* failsafe: stop, signal, return to home   */

#ifdef __cplusplus
}
#endif
#endif /* JS_HAL_H */
