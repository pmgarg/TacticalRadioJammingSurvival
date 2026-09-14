/* hal_esp32.c -- Mode H on real 2.4 GHz: ESP-NOW for the mesh, nRF24L01+ for the spectrum.
 *
 * NOT COMPILED IN CI: needs ESP-IDF v5.x headers. The interface it implements is exercised
 * on the desktop by hal_host.c + hal_selftest.c, so what is unproven here is the WIRING to
 * the radio, not the logic above it.
 *
 * Build: drop this into an ESP-IDF project's main/, add the RF24 driver, `idf.py build`.
 *
 * ---------------------------------------------------------------------------------------
 * WHY THE nRF24 IS NOT OPTIONAL
 *
 * The design's decisive features are S2 (noise-floor rise) and S5 (per-channel profile) --
 * FIDELITY.md showed S1 is largely unmeasurable because RSSI is only observed on frames
 * that decode. DESIGN section 11.2 assumed the ESP32 supplies S2 via rx_ctrl.noise_floor,
 * but that field reads 0 on several ESP-IDF versions (espressif/esp-idf#1751, #8022).
 *
 * So: probe it once at boot. If it is dead, hal_have_noise() returns false and the agent
 * takes S2 and S5 from the nRF24's RPD sweep instead -- rather than believing a zero, which
 * would read as "no attacker" on every jammed channel. A sensor that fails to 'all clear'
 * is worse than no sensor.
 *
 * WIRING (nRF24L01+ on VSPI; 3.3 V ONLY, and a 10 uF cap across VCC/GND):
 *   CE -> GPIO4   CSN -> GPIO5   SCK -> GPIO18   MOSI -> GPIO23   MISO -> GPIO19
 * ---------------------------------------------------------------------------------------
 */
#include "hal.h"

#include <string.h>

#if defined(ESP_PLATFORM)

#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nrf24.h"          /* any RPD-capable nRF24 driver */

#define BEACON_HZ        10
#define BEACON_PERIOD_US (1000000 / BEACON_HZ)
#define SHADOW_TAU_US    3000        /* reaction window: jammer delay + burst          */
#define RPD_SWEEPS       8           /* per channel, per scan                          */

/* nRF24 channel (1 MHz steps from 2400 MHz) at the centre of Wi-Fi channel `ch` */
#define WIFI_CH_TO_NRF(ch) ((uint8_t)(2407 + 5 * (ch) - 2400))

typedef struct { uint8_t src, seq, kind, slot, n_reports; } __attribute__((packed)) mesh_hdr_t;
typedef struct { uint8_t peer; int8_t rssi; uint8_t pdr_q; } __attribute__((packed)) link_report_t;

static struct
{
    uint8_t  me, channel, slot, n_slots;
    bool     tx_on, noise_ok;
    uint64_t tx_windows[32];         /* ring of recent TX timestamps, for S4'          */
    uint8_t  tx_w;
    uint64_t last_rx_us[HAL_MAX_PEERS];
    uint8_t  last_seq[HAL_MAX_PEERS];
    int8_t   rssi[HAL_MAX_PEERS];
    uint32_t rx_ok[HAL_MAX_PEERS], rx_exp[HAL_MAX_PEERS];
    float    rev_rssi[HAL_MAX_PEERS], rev_pdr[HAL_MAX_PEERS];
    uint64_t rev_t[HAL_MAX_PEERS];
    uint32_t sh_e, sh_m, si_e, si_m;
    uint32_t tx_att, tx_ack;
    float    noise_dbm;
    uint8_t  n_peers;
} H;

static bool in_tx_shadow(uint64_t t_us)
{
    for (int i = 0; i < 32; ++i)
    {
        uint64_t w = H.tx_windows[i];
        if (w && t_us + SHADOW_TAU_US >= w && w <= t_us + SHADOW_TAU_US) { return true; }
    }
    return false;
}

/* Promiscuous sniffer: RSSI and (maybe) the noise floor, per received frame. */
static void sniffer_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    (void)type;
    const wifi_promiscuous_pkt_t *p = (wifi_promiscuous_pkt_t *)buf;
    if (p->rx_ctrl.noise_floor != 0) { H.noise_dbm = (float)p->rx_ctrl.noise_floor; }
}

static void recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len)
{
    if (len < (int)sizeof(mesh_hdr_t)) { return; }
    const mesh_hdr_t *h = (const mesh_hdr_t *)data;
    if (h->src >= HAL_MAX_PEERS) { return; }
    uint64_t now = (uint64_t)esp_timer_get_time();

    H.rssi[h->src] = info->rx_ctrl ? (int8_t)info->rx_ctrl->rssi : -100;
    H.last_rx_us[h->src] = now;

    if (h->kind == 0)                    /* beacons only -- AUDIT F3c, the same bug     */
    {
        /* Beacons are periodic, so a sequence gap tells us exactly how many SHOULD have
         * arrived, and their due times. Bucket each miss by whether WE were transmitting
         * within tau of it: that is S4', and it needs only this node's own clock. */
        uint8_t gap = (uint8_t)(h->seq - H.last_seq[h->src]);
        for (uint8_t k = 1; k < gap; ++k)
        {
            uint64_t due = now - (uint64_t)k * BEACON_PERIOD_US;
            if (in_tx_shadow(due)) { H.sh_e++; H.sh_m++; } else { H.si_e++; H.si_m++; }
        }
        if (in_tx_shadow(now)) { H.sh_e++; } else { H.si_e++; }
        H.last_seq[h->src] = h->seq;
        H.rx_ok[h->src]++;
        H.rx_exp[h->src] += gap ? gap : 1;

        /* the peer's LinkReport about US -- the reverse link (TELEMETRY.md section 2) */
        const link_report_t *r = (const link_report_t *)(data + sizeof(mesh_hdr_t));
        for (uint8_t i = 0; i < h->n_reports &&
             (int)(sizeof(mesh_hdr_t) + (i + 1) * sizeof(link_report_t)) <= len; ++i)
        {
            if (r[i].peer == H.me)
            {
                H.rev_rssi[h->src] = (float)r[i].rssi;
                H.rev_pdr[h->src] = (float)r[i].pdr_q / 255.0f;
                H.rev_t[h->src] = now;
            }
        }
    }
}

static void send_cb(const uint8_t *mac, esp_now_send_status_t st)
{
    (void)mac;
    H.tx_att++;
    if (st == ESP_NOW_SEND_SUCCESS) { H.tx_ack++; }
    H.tx_windows[H.tx_w++ & 31] = (uint64_t)esp_timer_get_time();
}

int hal_init(uint8_t my_index, uint8_t channel)
{
    memset(&H, 0, sizeof H);
    H.me = my_index;
    H.channel = channel;
    H.tx_on = true;
    H.n_peers = HAL_MAX_PEERS;
    H.noise_dbm = 0.0f;

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    if (esp_wifi_init(&cfg) != ESP_OK) { return -1; }
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_start();
    esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_promiscuous_rx_cb(sniffer_cb);
    if (esp_now_init() != ESP_OK) { return -2; }
    esp_now_register_recv_cb(recv_cb);
    esp_now_register_send_cb(send_cb);

    nrf24_init();
    nrf24_set_rx_mode();

    /* Probe rx_ctrl.noise_floor once. If it never moves off zero, S2 comes from the
     * nRF24 and the agent is TOLD so, instead of reading a zero as "all clear". */
    vTaskDelay(pdMS_TO_TICKS(500));
    H.noise_ok = (H.noise_dbm != 0.0f);
    return 0;
}

bool hal_have_noise(void) { return H.noise_ok; }

void hal_sense(hal_obs_t *o)
{
    memset(o, 0, sizeof *o);
    o->t_us = (uint64_t)esp_timer_get_time();
    o->channel = H.channel;
    o->n_peers = H.n_peers;
    o->own_tx_active = H.tx_on;
    o->noise_floor_dbm = H.noise_ok ? H.noise_dbm : -96.0f;

    for (uint8_t i = 0; i < H.n_peers; ++i)
    {
        o->rssi_dbm[i] = (float)H.rssi[i];
        o->rx_ok[i] = H.rx_ok[i];
        o->rx_expected[i] = H.rx_exp[i] ? H.rx_exp[i] : 1;
        o->hb_gap_ms[i] = H.last_rx_us[i]
            ? (uint32_t)((o->t_us - H.last_rx_us[i]) / 1000u) : 60000u;
        o->rev_rssi_dbm[i] = H.rev_rssi[i];
        o->rev_pdr[i] = H.rev_pdr[i];
        o->rev_age_ms[i] = H.rev_t[i]
            ? (uint32_t)((o->t_us - H.rev_t[i]) / 1000u) : 60000u;
        H.rx_ok[i] = H.rx_exp[i] = 0;
    }
    o->tx_attempts = H.tx_att;  o->tx_acked = H.tx_ack;
    o->shadow_expected = H.sh_e; o->shadow_missed = H.sh_m;
    o->silent_expected = H.si_e; o->silent_missed = H.si_m;
    H.tx_att = H.tx_ack = H.sh_e = H.sh_m = H.si_e = H.si_m = 0;
}

int hal_scan(hal_scan_t *o, uint32_t dwell_ms)
{
    memset(o, 0, sizeof *o);
    o->t_us = (uint64_t)esp_timer_get_time();
    bool was = H.tx_on;
    hal_set_tx(false);                       /* a scan really does make us deaf         */

    for (uint8_t c = 0; c < HAL_N_CHANNELS; ++c)
    {
        uint8_t nrf = WIFI_CH_TO_NRF(c + 1);
        int hits = 0;
        for (int s = 0; s < RPD_SWEEPS; ++s)
        {
            /* RPD is one bit: "energy above about -64 dBm during the listen". Counting
             * hits over several dwells turns that bit into a usable energy estimate --
             * which is the whole trick that makes a 150-rupee part carry S5. */
            nrf24_set_channel(nrf);
            nrf24_start_listening();
            ets_delay_us(dwell_ms * 1000u / RPD_SWEEPS);
            hits += nrf24_test_rpd() ? 1 : 0;
            nrf24_stop_listening();
        }
        float frac = (float)hits / (float)RPD_SWEEPS;
        o->noise_dbm[c] = -100.0f + 35.0f * frac;   /* map occupancy onto the dBm scale */
        o->valid |= (uint8_t)(1u << c);
    }
    hal_set_tx(was);
    return 0;
}

int hal_neighbor_probe(uint8_t peer, uint32_t timeout_ms)
{
    if (peer >= H.n_peers) { return -1; }
    uint64_t before = H.last_rx_us[peer];
    mesh_hdr_t h = {.src = H.me, .seq = 0, .kind = 2, .slot = H.slot, .n_reports = 0};
    esp_now_send(NULL, (uint8_t *)&h, sizeof h);       /* kind 2 = directed probe       */
    for (uint32_t w = 0; w < timeout_ms; w += 5)
    {
        vTaskDelay(pdMS_TO_TICKS(5));
        if (H.last_rx_us[peer] != before) { return 1; }
    }
    return 0;
}

int hal_set_channel(uint8_t ch)
{
    if (ch < 1 || ch > HAL_N_CHANNELS) { return -1; }
    if (esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE) != ESP_OK) { return -2; }
    H.channel = ch;
    return 0;
}

int hal_set_tx_power(int8_t dbm)
{
    return esp_wifi_set_max_tx_power((int8_t)(dbm * 4)) == ESP_OK ? 0 : -1;  /* 0.25 dBm */
}

int hal_set_slot(uint8_t slot, uint8_t n_slots)
{
    /* Slots are a TRANSMIT GATE, enforced in the send path against the local monotonic
     * clock and re-synced on each beacon. A 100 ms frame with a 5 ms guard tolerates
     * about +/-2.5 ms of drift, which ESP32 crystals beat comfortably between beacons.
     * Recorded as a known sim-to-real divergence: ns-3 gives every node one perfect
     * global clock and no guard interval at all. */
    H.slot = slot;
    H.n_slots = n_slots;
    return 0;
}

int hal_set_tx(bool enabled) { H.tx_on = enabled; return 0; }

int hal_declare_lost(void)
{
    hal_set_tx(false);
    /* hand off to the flight controller's failsafe: return to home on the last known
     * good position. Beyond the scope of the radio agent. */
    return 0;
}

#endif /* ESP_PLATFORM */
