/*
 * Jamming Survival — ns-3 world (Tier 1, authoritative).
 *
 * Arbitrary mesh from a flat key=value scenario config (capstone/sim/export_ns3.py),
 * stock OLSRv1 over SpectrumWifiPhy on a shared MultiModelSpectrumChannel, with real
 * RF interference from WaveformGenerator jammers (barrage / spot / sweep / reactive).
 *
 * Measurement plane (MeshNodeApp) gives TRUE application-level, per-link delivery from
 * sequence-numbered beacons -- not a PHY proxy. Per-link RSSI is attributed by parsing
 * the 802.11 transmitter address. OLSR route state and TDMA slotting are verified.
 *
 * Outputs per 100 ms:
 *   <out>.percept.csv  what the agent may see
 *   <out>.truth.csv    verifier only
 *   <out>.routes.csv   OLSR route table size / hop counts (protocol verification)
 */
#include "mesh-node-app.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/ism-spectrum-value-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/network-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/olsr-routing-protocol.h"
#include "ns3/propagation-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("JammingSim");

// --------------------------------------------------------------------------- //
class Cfg
{
  public:
    bool Load(const std::string& p)
    {
        std::ifstream f(p);
        if (!f)
        {
            return false;
        }
        std::string l;
        while (std::getline(f, l))
        {
            auto e = l.find('=');
            if (e != std::string::npos)
            {
                m[l.substr(0, e)] = l.substr(e + 1);
            }
        }
        return true;
    }

    std::string S(const std::string& k, const std::string& d = "") const
    {
        auto i = m.find(k);
        return i == m.end() ? d : i->second;
    }

    double D(const std::string& k, double d = 0) const
    {
        auto i = m.find(k);
        return i == m.end() ? d : std::atof(i->second.c_str());
    }

    int I(const std::string& k, int d = 0) const
    {
        auto i = m.find(k);
        return i == m.end() ? d : std::atoi(i->second.c_str());
    }

  private:
    std::map<std::string, std::string> m;
};

static std::string
K(const std::string& p, int i, const std::string& s)
{
    std::ostringstream o;
    o << p << "." << i << "." << s;
    return o.str();
}

// --------------------------------------------------------------------------- //
struct JammerSpec
{
    std::string id, type;
    double eirp{15}, duty{1}, dwellMs{200}, delayUs{10}, burstMs{1.5}, pFire{1};
    std::vector<int> channels;
    Ptr<WaveformGenerator> wg;
    bool on{false};
    int curIdx{0};
    int psdCh{-1};  // AUDIT F3a: channel the PSD is currently pointed at (-1 = unset)
    // ROBUSTNESS (capstone): re-entrancy guard for ReactiveTrigger. True from the
    // moment a burst is scheduled until its Stop() fires -- see ReactiveTrigger.
    bool burstActive{false};
};

static std::vector<JammerSpec> g_jam;
static SpectrumValue5MhzFactory g_psd;
static uint32_t g_nCh = 8;
static uint32_t g_agentIdx = 0;
static int g_curCh = 6;


/* AUDIT F3f (MEASURED, then fixed): ns-3's SpectrumValue5MhzFactory puts the four
 * full-power bins of CreateTxPowerSpectralDensity(p, ch) at model indices ch+3..ch+6.
 * On that model, band index k spans [2387 + 5k, 2392 + 5k] MHz, so those four bins form a
 * 20 MHz block centred at 2412 + 5*ch MHz -- while 802.11 channel `ch` is centred at
 * 2407 + 5*ch MHz (ch1 = 2412). The helper therefore radiates EXACTLY ONE CHANNEL HIGH.
 *
 * Measured, not assumed: with jam.0.ch.0=6 the hottest band_power bin was channel 7
 * (capstone/tests/test_world_ns3.py). The scenarios still "worked" because 20 MHz channels
 * overlap heavily, which is why this hid -- but S5's hot-channel IDENTITY was shifted, so
 * hop_channel was picking its "cleanest" channel from a shifted map.
 *
 * Shifting the helper's argument by -1 would fix it for ch >= 2 and write index -1 for
 * ch == 1, so the block is built here instead, with the same 802.11 transmit mask
 * (-28 dB and -40 dB shoulders) and bounds-checked indices.
 */
static Ptr<SpectrumValue>
JamPsd(double watts, int wifiCh)
{
    Ptr<SpectrumValue> psd = g_psd.CreateTxPowerSpectralDensity(watts, 1);
    (*psd) = 0.0;                       // reuse the 5 MHz model, start from silence
    const double d = watts / 20e6;      // density over the 20 MHz block
    const int n = (int)psd->GetSpectrumModel()->GetNumBands();
    auto put = [&](int k, double v) {
        if (k >= 0 && k < n)
        {
            (*psd)[k] = v;
        }
    };
    // one bin lower than the stock helper at every position -- see the arithmetic above
    put(wifiCh - 2, d * 1e-4);          // -40 dB
    put(wifiCh - 1, d * 1e-4);
    put(wifiCh + 0, d * 0.0015849);     // -28 dB
    put(wifiCh + 1, d * 0.0015849);
    put(wifiCh + 2, d);                 // the 20 MHz block, centred on 2407 + 5*ch MHz
    put(wifiCh + 3, d);
    put(wifiCh + 4, d);
    put(wifiCh + 5, d);
    put(wifiCh + 6, d * 0.0015849);     // -28 dB
    put(wifiCh + 7, d * 0.0015849);
    put(wifiCh + 8, d * 1e-4);          // -40 dB
    put(wifiCh + 9, d * 1e-4);
    return psd;
}

static void
SetCh(JammerSpec& j, int ch)
{
    j.wg->SetTxPowerSpectralDensity(JamPsd(std::pow(10.0, (j.eirp - 30) / 10.0), ch));
    j.psdCh = ch;
}

static void
SetWide(JammerSpec& j)
{
    Ptr<SpectrumValue> acc;
    double per = std::pow(10.0, (j.eirp - 30) / 10.0) / std::max<size_t>(1, j.channels.size());
    for (int c : j.channels)
    {
        Ptr<SpectrumValue> p = JamPsd(per, c);
        if (!acc)
        {
            acc = p->Copy();
        }
        else
        {
            (*acc) += (*p);
        }
    }
    if (acc)
    {
        j.wg->SetTxPowerSpectralDensity(acc);
    }
}

static void
JamOn(size_t i)
{
    if (i >= g_jam.size())
    {
        return;
    }
    auto& j = g_jam[i];
    j.on = true;
    if (j.type == "barrage")
    {
        SetWide(j);
    }
    else if (!j.channels.empty())
    {
        SetCh(j, j.channels[0]);
    }
    j.wg->Start();
}

static void
JamOff(size_t i)
{
    if (i < g_jam.size())
    {
        g_jam[i].on = false;
        g_jam[i].wg->Stop();
    }
}

/* AUDIT F3a(a): a reactive jammer is a follower -- it jams whatever channel it just
 * heard the victim transmit on. The config gives it no channel of its own
 * (scenarios set `channels: []`, which export_ns3.py expands to the whole 1..8 band),
 * so jam.i.ch.0 == 1 and the construction-time PSD sat on channel 1 while the mesh
 * ran on channel 6 -- the burst never touched the victim. Point the PSD at the
 * channel the mesh is actually using, unless exactly one channel was configured
 * explicitly, in which case honour it. */
static void
PointAtVictim(JammerSpec& j)
{
    int ch = (j.channels.size() == 1) ? j.channels[0] : g_curCh;
    if (j.psdCh != ch)
    {
        SetCh(j, ch);
    }
}

/* AUDIT F3a(a)+(b): arm a reactive jammer. Same PSD setup JamOn() does for every
 * other jammer type, but WITHOUT wg->Start() -- a reactive jammer only radiates from
 * ReactiveTrigger(). Scheduled at the event's onset time so there is a clean
 * pre-onset baseline (it used to be armed during main() setup, i.e. from t=0). */
static void
JamArmReactive(size_t i)
{
    if (i >= g_jam.size())
    {
        return;
    }
    auto& j = g_jam[i];
    j.on = true;
    PointAtVictim(j);
}

static void
Sweep(size_t i)
{
    if (i >= g_jam.size())
    {
        return;
    }
    auto& j = g_jam[i];
    if (j.on && !j.channels.empty())
    {
        j.curIdx = (j.curIdx + 1) % (int)j.channels.size();
        SetCh(j, j.channels[j.curIdx]);
    }
    Simulator::Schedule(MilliSeconds(j.dwellMs), &Sweep, i);
}

/* Reactive: fire a burst when the agent starts transmitting. Hooked to PhyTxBegin,
 * which is the honest analogue of an energy-detecting attacker -- it reacts to OUR
 * transmission, within microseconds, and only then. */
static Ptr<UniformRandomVariable> g_rand;

static void
ReactiveTrigger(Ptr<const Packet>, double)
{
    for (size_t i = 0; i < g_jam.size(); ++i)
    {
        auto& j = g_jam[i];
        if (j.type != "reactive" || !j.on)
        {
            continue;
        }
        if (g_rand->GetValue() > j.pFire)
        {
            continue;
        }
        // ROBUSTNESS (capstone): a re-entrancy guard. Without it, two agent
        // transmissions closer together than delayUs + burstMs each schedule their
        // own Start()/Stop() pair on the SAME WaveformGenerator, and the overlapping
        // pair can call Start() while a previous burst is still running or Stop() out
        // of order. That leaves SpectrumWifiPhy's preamble-reception state
        // inconsistent and trips ns-3's own "!m_wifiPhy->m_currentEvent" assert in
        // phy-entity.cc -- reproduced on reactive_on_tx: NS_ASSERT failed at
        // +15.7s and +34.1s, both mid-episode, both this exact condition. A second
        // TX arriving mid-burst is real (retries, multiple nodes) and must not start
        // a second overlapping burst; skip it rather than corrupt the PHY's state.
        if (j.burstActive)
        {
            continue;
        }
        // AUDIT F3a(a): re-point the PSD on every trigger, so a mesh-wide
        // hop_channel does not make the follower jam an empty channel.
        PointAtVictim(j);
        j.burstActive = true;
        size_t idx = i;
        Simulator::Schedule(MicroSeconds(j.delayUs), [idx]() {
            if (idx < g_jam.size() && g_jam[idx].on)
            {
                g_jam[idx].wg->Start();
            }
        });
        Simulator::Schedule(MicroSeconds(j.delayUs) + MilliSeconds(j.burstMs), [idx]() {
            if (idx < g_jam.size())
            {
                g_jam[idx].wg->Stop();
                g_jam[idx].burstActive = false;
            }
        });
    }
}

// --------------------------------------------------------------------------- //
// Measurement
// --------------------------------------------------------------------------- //
static std::map<Mac48Address, uint32_t> g_macToIdx;
static std::map<uint32_t, double> g_linkRssi;   // per source node index
static std::map<uint32_t, double> g_linkRssiT;  // when it was measured
static std::vector<double> g_band, g_bandFloor, g_bandFloorBridge;
/* MEAN linear power per channel over the clean (agent-silent) analyzer reports in the
 * current window, for the CSV path and the bridge path separately.
 *
 * WHY THIS EXISTS. The band was reported as a strict MIN-HOLD over the window. Min-hold
 * is the right idea for "the noise floor when nobody is transmitting", and it correctly
 * rejects our own bursts -- but it makes any DUTY-CYCLED jammer invisible, because the
 * minimum always lands in one of the jammer's own gaps. The corpus runs spot/reactive at
 * duty 0.7; the hand-written world-gate scenarios run at duty 1.0. So the gate passed
 * 17/17 while every duty-cycled jammer in the actual corpus was reported at the quiet
 * floor, and S2 (noise_delta_base) and S5 (scan_noise_spread) were identically zero for
 * whole episodes.
 *
 * Mean power over the agent-silent samples is what an ENERGY DETECTOR measures, it is
 * monotone in jammer duty, and it is what the hardware path does too (hal_esp32.c counts
 * nRF24 RPD hits and maps the hit fraction onto a dBm scale). meas_floor_dbm keeps the
 * min-hold, so "the floor" and "the energy" are now two different numbers instead of one
 * number that could only ever answer the first question. */
static std::vector<std::vector<double>> g_bandSamp, g_bandSampBridge;
static double g_lastNoise = -96.0;
/* The analyzer sits at the agent, so while the agent transmits it measures the
 * agent's own 16 dBm signal, not the channel. A real radio cannot listen while it
 * transmits either. Gate the floor estimate on TX state -- this is what makes the
 * S2 (noise rise) and S3 (energy without preamble) statistics honest in ns-3. */
// PHY-measured RSSI per (receiving node, transmitting node). This is the
// "PHY attaches the measurement, MAC attributes it to a source" step; MeshNodeApp
// then echoes it in its beacons so neighbours learn their reverse link.
static std::vector<std::map<uint32_t, double>>* g_nodeRssi = nullptr;
static std::vector<Ptr<MeshNodeApp>>* g_apps = nullptr;
// recent agent TX intervals, for the TX-shadow statistic
static std::deque<std::pair<double, double>> g_txIntervals;
static double g_txOpenedAt = -1.0;
static uint64_t g_txAttempts = 0, g_txFinalFail = 0;
static double g_deferAccum = 0.0;
static uint32_t g_deferN = 0;
static double g_lastEnqueue = -1.0;
static bool g_agentTx = false;
static bool g_txSinceReport = false;   // any agent TX overlapping this report window
static uint64_t g_txBegin = 0, g_cleanReports = 0, g_dirtyReports = 0;
static uint64_t g_rxOk = 0, g_rxErr = 0;

/* superseded FIXME(F3f), kept for the record: suspected one-channel offset between this
 * helper and SpectrumValue5MhzFactory. ChHz(ch) is the true 802.11b/g centre of
 * channel `ch` (ch1 = 2412 MHz), but SpectrumValue5MhzFactory numbers its 5 MHz
 * bands from 2402 MHz with a 0-based index, so
 * CreateTxPowerSpectralDensity(power, ch) -- called with a 1-based WiFi channel
 * number in SetCh()/SetWide()/the jammer install site -- may land one 5 MHz band
 * (i.e. one channel) above/below the channel the mesh is actually on. Verify by
 * running a single spot jammer on a known channel and checking which g_band[] bin
 * lights up in <out>.percept.csv; if it is off by one, the fix is an index
 * adjustment at every CreateTxPowerSpectralDensity() call site, NOT here.
 * Deliberately left unchanged in this pass -- needs measurement, not a guess. */
static double
ChHz(int ch)
{
    return (2412.0 + 5.0 * (ch - 1)) * 1e6;
}

static void
AnalyzerReport(Ptr<const SpectrumValue> psd)
{
    if (!psd)
    {
        return;
    }
    Ptr<const SpectrumModel> mo = psd->GetSpectrumModel();
    g_band.assign(g_nCh, -200.0);
    for (uint32_t c = 0; c < g_nCh; ++c)
    {
        double lo = ChHz(c + 1) - 10e6, hi = ChHz(c + 1) + 10e6, w = 0;
        uint32_t i = 0;
        for (auto b = mo->Begin(); b != mo->End(); ++b, ++i)
        {
            double fc = (b->fl + b->fh) / 2.0;
            if (fc >= lo && fc <= hi)
            {
                w += (*psd)[i] * (b->fh - b->fl);
            }
        }
        g_band[c] = (w > 0) ? 10.0 * std::log10(w * 1000.0) : -200.0;
    }
    if (g_bandFloor.size() != g_nCh)
    {
        g_bandFloor.assign(g_nCh, 1e9);
    }
    /* An analyzer report averages over its whole Resolution window, so it is only
     * usable if the agent was silent for ALL of that window -- checking the
     * instantaneous TX flag at report time is not enough. */
    bool dirty = g_agentTx || g_txSinceReport;
    g_txSinceReport = false;
    if (dirty)
    {
        g_dirtyReports++;
        return;
    }
    g_cleanReports++;
    if (g_bandFloorBridge.size() != g_nCh)
    {
        g_bandFloorBridge.assign(g_nCh, 1e9);
    }
    if (g_bandSamp.size() != g_nCh)
    {
        g_bandSamp.assign(g_nCh, {});
    }
    if (g_bandSampBridge.size() != g_nCh)
    {
        g_bandSampBridge.assign(g_nCh, {});
    }
    for (uint32_t c = 0; c < g_nCh; ++c)
    {
        g_bandFloor[c] = std::min(g_bandFloor[c], g_band[c]);
        g_bandFloorBridge[c] = std::min(g_bandFloorBridge[c], g_band[c]);
        g_bandSamp[c].push_back(g_band[c]);
        g_bandSampBridge[c].push_back(g_band[c]);
    }
}

/* MEDIAN power per channel over this window's agent-silent reports, in dBm.
 *
 * Neither extreme works. The strict MINIMUM rejects every transmission, including a
 * duty-cycled jammer's -- the minimum always lands in one of its gaps, so a duty-0.7 spot
 * jammer reads exactly the quiet floor and S2/S5 are identically zero. The MEAN rejects
 * nothing: the other five nodes beacon and carry data continuously, so the mean sits tens
 * of dB above the floor before any jammer exists at all.
 *
 * The median is the level exceeded half the time. Ordinary mesh traffic occupies well
 * under half of any window, so it does not move the median; a jammer at duty >= 0.5 does,
 * by construction. That is the statistic an energy detector with a sensible threshold
 * actually implements, and it is what the nRF24 RPD hit-count approximates on hardware.
 *
 * Falls back to the min-hold when the window held no agent-silent report at all. */
static std::vector<double>
BandEnergy(std::vector<std::vector<double>>& samp, const std::vector<double>& fallback)
{
    std::vector<double> out(g_nCh, -200.0);
    for (uint32_t c = 0; c < g_nCh; ++c)
    {
        if (c < samp.size() && !samp[c].empty())
        {
            std::vector<double>& v = samp[c];
            const size_t mid = v.size() / 2;
            std::nth_element(v.begin(), v.begin() + mid, v.end());
            out[c] = v[mid];
        }
        else if (c < fallback.size() && fallback[c] < 1e8)
        {
            out[c] = fallback[c];
        }
    }
    return out;
}

static void
AgentTxBegin(Ptr<const Packet>, double)
{
    g_agentTx = true;
    g_txSinceReport = true;
    g_txBegin++;
}

static void
AgentTxEnd(Ptr<const Packet>)
{
    g_agentTx = false;
}

/* Per-link RSSI: parse the 802.11 transmitter address so a signal level can be
 * attributed to the peer that sent it. Aggregate RSSI cannot distinguish "one peer
 * faded" from "the whole channel died" -- that distinction is the S6 statistic. */
static void
SnifferRx(Ptr<const Packet> p,
          uint16_t,
          WifiTxVector,
          MpduInfo,
          SignalNoiseDbm sn,
          uint16_t)
{
    g_lastNoise = sn.noise;
    g_rxOk++;
    Ptr<Packet> c = p->Copy();
    WifiMacHeader h;
    if (c->PeekHeader(h))
    {
        Mac48Address a2 = h.GetAddr2();
        auto it = g_macToIdx.find(a2);
        if (it != g_macToIdx.end())
        {
            g_linkRssi[it->second] = sn.signal;
            g_linkRssiT[it->second] = Simulator::Now().GetSeconds();
        }
    }
}

/** Sniffer installed on EVERY node: record the PHY-measured RSSI for the transmitter
 *  of each received frame. Same information an ESP32 gets from rx_ctrl. */
static void
NodeSnifferRx(uint32_t selfIdx,
              Ptr<const Packet> p,
              uint16_t,
              WifiTxVector,
              MpduInfo,
              SignalNoiseDbm sn,
              uint16_t)
{
    if (!g_nodeRssi || selfIdx >= g_nodeRssi->size())
    {
        return;
    }
    Ptr<Packet> c = p->Copy();
    WifiMacHeader h;
    if (!c->PeekHeader(h))
    {
        return;
    }
    auto it = g_macToIdx.find(h.GetAddr2());
    if (it == g_macToIdx.end())
    {
        return;
    }
    (*g_nodeRssi)[selfIdx][it->second] = sn.signal;
    if (g_apps && selfIdx < g_apps->size() && (*g_apps)[selfIdx])
    {
        (*g_apps)[selfIdx]->RxRssi()[it->second] = sn.signal;
    }
}

/** Agent TX interval tracking -> feeds both the analyzer gate and the TX-shadow test. */
static void
AgentTxBeginKpi(Ptr<const Packet>, double)
{
    g_txOpenedAt = Simulator::Now().GetSeconds();
    g_txAttempts++;
    if (g_lastEnqueue >= 0.0)
    {
        // enqueue -> on-air delay == how long CSMA had to wait for decodable carrier
        g_deferAccum += std::max(0.0, (g_txOpenedAt - g_lastEnqueue)) * 1000.0;
        g_deferN++;
        g_lastEnqueue = -1.0;
    }
}

static void
AgentTxEndKpi(Ptr<const Packet>)
{
    if (g_txOpenedAt >= 0.0)
    {
        g_txIntervals.emplace_back(g_txOpenedAt, Simulator::Now().GetSeconds());
        while (g_txIntervals.size() > 4000)
        {
            g_txIntervals.pop_front();
        }
        g_txOpenedAt = -1.0;
    }
}

static void
MacFinalFail(Mac48Address)
{
    g_txFinalFail++;
}

/** Was the agent transmitting within tau of time t? (TELEMETRY.md §1) */
static bool
InTxShadow(double t)
{
    const double tau = 0.003;
    for (auto it = g_txIntervals.rbegin(); it != g_txIntervals.rend(); ++it)
    {
        if (it->second + tau < t - 0.5)
        {
            break; // older than we care about
        }
        if (t >= it->first - tau && t <= it->second + tau)
        {
            return true;
        }
    }
    return false;
}

static void
PhyRxDrop(Ptr<const Packet>, WifiPhyRxfailureReason)
{
    g_rxErr++;
}

/* AUDIT F3g: the spectrum analyzer is its own node and used to be pinned with a
 * ConstantPositionMobilityModel at the agent's INITIAL coordinates. After a `move`
 * action, or in any scenario with mobility, the noise-floor / band-power reading
 * therefore came from a point the agent had left. Keep the analyzer co-located with
 * the agent instead. (The agent's MobilityModel is not simply aggregated onto the
 * analyzer node: AggregateObject() would merge the two nodes' whole aggregates.) */
static uint32_t g_flowSrcIdx = 0;   // AUDIT: node index of flow.0.src
static Ptr<MobilityModel> g_agentMob, g_analyzerMob;

static void
SyncAnalyzerPos()
{
    if (g_agentMob && g_analyzerMob)
    {
        g_analyzerMob->SetPosition(g_agentMob->GetPosition());
    }
}

static void
SyncAnalyzerPosLoop()
{
    SyncAnalyzerPos();
    // 100 ms == the sampling period, so the analyzer is in the right place for every
    // reported sample even under ConstantVelocity / waypoint mobility.
    Simulator::Schedule(MilliSeconds(100), &SyncAnalyzerPosLoop);
}

/* AUDIT F3b: the CSV sampler (`tick`) and the live bridge (`decide`) are BOTH
 * scheduled at t=1.0 s and both re-arm every 100 ms, so they fire at identical
 * simulation times. Whichever ran first read MeshNodeApp's TX-shadow accumulators and
 * then called ResetShadow(); the other one read zeros and reset again -- which is why
 * shadow_exp/shadow_miss/silent_exp/silent_miss were always 0 in live bridge runs.
 * Read and reset the accumulator exactly ONCE per simulation instant and cache the
 * values for whoever reads second. Offline (CSV-only) runs are unaffected: `tick` is
 * then the sole caller, so it still gets one read+reset per 100 ms window. */
struct ShadowSample
{
    int64_t ts{-1};  // Simulator::Now() in time steps; -1 = nothing sampled yet
    uint32_t shE{0}, shM{0}, siE{0}, siM{0};
};

static ShadowSample g_shadowSample;

static const ShadowSample&
SampleShadow(Ptr<MeshNodeApp> me)
{
    int64_t now = Simulator::Now().GetTimeStep();
    if (g_shadowSample.ts != now)
    {
        g_shadowSample.ts = now;
        g_shadowSample.shE = me->ShadowExpected();
        g_shadowSample.shM = me->ShadowMissed();
        g_shadowSample.siE = me->SilentExpected();
        g_shadowSample.siM = me->SilentMissed();
        me->ResetShadow();
    }
    return g_shadowSample;
}

// --------------------------------------------------------------------------- //
// AgentBridge: line-delimited JSON over a UNIX socket.
//
// At each decision epoch the simulator BLOCKS on a read from the socket. ns-3 is
// single threaded, so a blocking read simply stalls wall-clock time while simulation
// time stays frozen -- the agent can take as long as it likes without distorting the
// experiment. Costs are charged by actually suppressing TX/RX for the action's
// duration, so a spectrum scan really does make the drone deaf.
// --------------------------------------------------------------------------- //
static int g_sock = -1;
static std::string g_rxbuf;

static bool
BridgeConnect(const std::string& path)
{
    g_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_sock < 0)
    {
        return false;
    }
    struct sockaddr_un a;
    memset(&a, 0, sizeof(a));
    a.sun_family = AF_UNIX;
    strncpy(a.sun_path, path.c_str(), sizeof(a.sun_path) - 1);
    if (connect(g_sock, (struct sockaddr*)&a, sizeof(a)) < 0)
    {
        close(g_sock);
        g_sock = -1;
        return false;
    }
    return true;
}

static bool
BridgeSend(const std::string& line)
{
    if (g_sock < 0)
    {
        return false;
    }
    std::string l = line + "\n";
    size_t off = 0;
    while (off < l.size())
    {
        ssize_t n = write(g_sock, l.data() + off, l.size() - off);
        if (n <= 0)
        {
            return false;
        }
        off += n;
    }
    return true;
}

static bool
BridgeRecv(std::string& out)
{
    while (true)
    {
        auto nl = g_rxbuf.find('\n');
        if (nl != std::string::npos)
        {
            out = g_rxbuf.substr(0, nl);
            g_rxbuf.erase(0, nl + 1);
            return true;
        }
        char buf[4096];
        ssize_t n = read(g_sock, buf, sizeof(buf));
        if (n <= 0)
        {
            return false;
        }
        g_rxbuf.append(buf, n);
    }
}

/** Minimal extractor for the small, fixed-shape reply we control. */
static std::string
JGet(const std::string& j, const std::string& key)
{
    std::string k = "\"" + key + "\"";
    auto p = j.find(k);
    if (p == std::string::npos)
    {
        return "";
    }
    p = j.find(':', p + k.size());
    if (p == std::string::npos)
    {
        return "";
    }
    ++p;
    while (p < j.size() && isspace((unsigned char)j[p]))
    {
        ++p;
    }
    if (p < j.size() && j[p] == '"')
    {
        auto e = j.find('"', p + 1);
        return j.substr(p + 1, e - p - 1);
    }
    auto e = j.find_first_of(",}", p);
    return j.substr(p, e - p);
}

static double
JNum(const std::string& j, const std::string& key, double d = 0.0)
{
    std::string v = JGet(j, key);
    return v.empty() ? d : std::atof(v.c_str());
}

// --------------------------------------------------------------------------- //
int
main(int argc, char* argv[])
{
    std::string cfgPath, out = "run";
    uint32_t tdmaSlots = 0;
    std::string sockPath;
    CommandLine cmd(__FILE__);
    cmd.AddValue("config", "scenario config", cfgPath);
    cmd.AddValue("out", "output prefix", out);
    cmd.AddValue("tdmaSlots", "TDMA slots (0/1 = CSMA only)", tdmaSlots);
    cmd.AddValue("sock", "UNIX socket for the live agent bridge (empty = offline)", sockPath);
    cmd.Parse(argc, argv);

    Cfg C;
    if (cfgPath.empty() || !C.Load(cfgPath))
    {
        std::cerr << "ERROR: --config required\n";
        return 1;
    }

    const double dur = C.D("duration", 60.0);
    g_nCh = C.I("n_channels", 8);
    const int ch0 = C.I("channel", 6);
    g_curCh = ch0;
    RngSeedManager::SetSeed(C.I("seed", 1));
    RngSeedManager::SetRun(1);
    g_rand = CreateObject<UniformRandomVariable>();

    const int N = C.I("n_nodes", 0);
    NodeContainer nodes;
    nodes.Create(N);
    std::vector<std::string> ids(N), roles(N);
    for (int i = 0; i < N; ++i)
    {
        ids[i] = C.S(K("node", i, "id"));
        roles[i] = C.S(K("node", i, "role"), "peer");
        if (roles[i] == "agent")
        {
            g_agentIdx = i;
        }
    }

    MobilityHelper mob;
    Ptr<ListPositionAllocator> pa = CreateObject<ListPositionAllocator>();
    for (int i = 0; i < N; ++i)
    {
        pa->Add(Vector(C.D(K("node", i, "x")), C.D(K("node", i, "y")), C.D(K("node", i, "z"))));
    }
    mob.SetPositionAllocator(pa);
    mob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    mob.Install(nodes);

    for (int i = 0, nm = C.I("n_mobility", 0); i < nm; ++i)
    {
        std::string who = C.S(K("mob", i, "node"));
        for (int k = 0; k < N; ++k)
        {
            if (ids[k] != who)
            {
                continue;
            }
            auto cv = nodes.Get(k)->GetObject<ConstantVelocityMobilityModel>();
            std::string t = C.S(K("mob", i, "type"), "static");
            if (t == "constant_velocity")
            {
                cv->SetVelocity(
                    Vector(C.D(K("mob", i, "vx")), C.D(K("mob", i, "vy")), C.D(K("mob", i, "vz"))));
            }
            else if (t == "waypoint" && C.I(K("mob", i, "n_wp")) > 0)
            {
                double sp = C.D(K("mob", i, "speed"), 5.0);
                std::ostringstream b;
                b << "mob." << i << ".wp.0.";
                Vector p0 = cv->GetPosition();
                Vector d(C.D(b.str() + "x") - p0.x,
                         C.D(b.str() + "y") - p0.y,
                         C.D(b.str() + "z") - p0.z);
                double n = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
                if (n > 1e-6)
                {
                    cv->SetVelocity(Vector(d.x / n * sp, d.y / n * sp, d.z / n * sp));
                }
            }
        }
    }

    // ---------------- spectrum channel + PHY ---------------- //
    Ptr<MultiModelSpectrumChannel> sc = CreateObject<MultiModelSpectrumChannel>();
    Ptr<LogDistancePropagationLossModel> pl = CreateObject<LogDistancePropagationLossModel>();
    pl->SetAttribute("Exponent", DoubleValue(C.D("exponent", 2.4)));
    pl->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    pl->SetAttribute("ReferenceLoss", DoubleValue(C.D("ref_loss", 40.0)));
    Ptr<NakagamiPropagationLossModel> g_nak;
    if (C.S("fading", "none") == "nakagami")
    {
        Ptr<NakagamiPropagationLossModel> nk = CreateObject<NakagamiPropagationLossModel>();
        g_nak = nk;
        nk->SetAttribute("m0", DoubleValue(C.D("nakagami_m", 1.0)));
        nk->SetAttribute("m1", DoubleValue(C.D("nakagami_m", 1.0)));
        nk->SetAttribute("m2", DoubleValue(C.D("nakagami_m", 1.0)));
        pl->SetNext(nk);
    }
    sc->AddPropagationLossModel(pl);
    sc->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    SpectrumWifiPhyHelper phy;
    phy.SetChannel(sc);
    double txp = C.D("node.0.txpower", 16.0);
    // AUDIT: current agent TX power, tracked across set_tx_power so the bridge state
    // reports what the radio is actually doing rather than the config value.
    double curTxp = txp;
    phy.Set("TxPowerStart", DoubleValue(txp));
    phy.Set("TxPowerEnd", DoubleValue(txp));
    phy.Set("RxNoiseFigure", DoubleValue(7.0));
    {
        std::ostringstream s;
        s << "{" << ch0 << ", 20, BAND_2_4GHZ, 0}";
        phy.Set("ChannelSettings", StringValue(s.str()));
    }

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("HtMcs2"),
                                 "ControlMode",
                                 StringValue("HtMcs0"));
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devs = wifi.Install(phy, mac, nodes);

    for (int i = 0; i < N; ++i)
    {
        Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devs.Get(i));
        g_macToIdx[Mac48Address::ConvertFrom(wd->GetAddress())] = i;
    }

    OlsrHelper olsr;
    Ipv4ListRoutingHelper lr;
    lr.Add(olsr, 10);
    InternetStackHelper stack;
    stack.SetRoutingHelper(lr);
    stack.Install(nodes);
    Ipv4AddressHelper ah;
    ah.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ifs = ah.Assign(devs);

    // ---------------- measurement / traffic plane ---------------- //
    const uint16_t PORT = 9999;
    // Limited broadcast: a subnet-directed address (10.1.1.255) has no route installed
    // by OLSR, so packets to it are silently dropped. 255.255.255.255 is link-local.
    Ipv4Address bcast = Ipv4Address::GetBroadcast();
    std::vector<Ptr<MeshNodeApp>> apps(N);
    // primary mission flow from the config (first flow wins as "the" data flow)
    std::string fsrc = C.S("flow.0.src"), fdst = C.S("flow.0.dst");
    double fkbps = C.D("flow.0.kbps", 200.0);
    uint32_t fbytes = (uint32_t)C.D("flow.0.bytes", 512);
    int dstIdx = 0;
    for (int k = 0; k < N; ++k)
    {
        if (ids[k] == fdst)
        {
            dstIdx = k;
        }
        if (ids[k] == fsrc)
        {
            g_flowSrcIdx = (uint32_t)k;   // AUDIT: data_tx is reported from THIS node
        }
    }

    for (int i = 0; i < N; ++i)
    {
        Ptr<MeshNodeApp> a = CreateObject<MeshNodeApp>();
        bool hasData = (ids[i] == fsrc);
        a->Setup(i,
                 bcast,
                 PORT,
                 MilliSeconds(100),
                 hasData ? fkbps : 0.0,
                 fbytes,
                 ifs.GetAddress(dstIdx),
                 hasData);
        if (hasData)
        {
            // AUDIT F3d: the mission flow now honours flow.0.start / flow.0.stop,
            // exactly as background flows (index >= 1) already do below.
            a->SetDataWindow(Seconds(C.D("flow.0.start", 0.0)),
                             Seconds(C.D("flow.0.stop", dur)));
        }
        if (tdmaSlots > 1)
        {
            a->SetTdma(i % tdmaSlots, tdmaSlots, MilliSeconds(100));
        }
        nodes.Get(i)->AddApplication(a);
        a->SetStartTime(Seconds(0.5));
        a->SetStopTime(Seconds(dur));
        apps[i] = a;
    }

    // ---- telemetry wiring: per-node PHY sniffers, shadow query, TX KPIs ----
    static std::vector<std::map<uint32_t, double>> s_nodeRssi;
    s_nodeRssi.assign(N, {});
    g_nodeRssi = &s_nodeRssi;
    static std::vector<Ptr<MeshNodeApp>> s_apps;
    s_apps = apps;
    g_apps = &s_apps;
    for (int i = 0; i < N; ++i)
    {
        Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devs.Get(i));
        uint32_t idx = i;
        wd->GetPhy()->TraceConnectWithoutContext(
            "MonitorSnifferRx",
            MakeCallback(&NodeSnifferRx).Bind(idx));
        apps[i]->SetTxShadowQuery(&InTxShadow);
        if ((uint32_t)i == g_agentIdx)
        {
            apps[i]->SetOnSend([]() { g_lastEnqueue = Simulator::Now().GetSeconds(); });
        }
    }

    // extra background/congestion flows (index >= 1)
    for (int i = 1, nf = C.I("n_flows", 0); i < nf; ++i)
    {
        std::string s = C.S(K("flow", i, "src")), d = C.S(K("flow", i, "dst"));
        int si = -1, di = -1;
        for (int k = 0; k < N; ++k)
        {
            if (ids[k] == s)
            {
                si = k;
            }
            if (ids[k] == d)
            {
                di = k;
            }
        }
        if (si < 0 || di < 0)
        {
            continue;
        }
        uint16_t p2 = 9100 + i;
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), p2));
        auto sa = sink.Install(nodes.Get(di));
        sa.Start(Seconds(0.5));
        sa.Stop(Seconds(dur));
        OnOffHelper on("ns3::UdpSocketFactory", InetSocketAddress(ifs.GetAddress(di), p2));
        on.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        on.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        on.SetAttribute("DataRate",
                        DataRateValue(DataRate((uint64_t)(C.D(K("flow", i, "kbps"), 20) * 1000))));
        on.SetAttribute("PacketSize", UintegerValue((uint32_t)C.D(K("flow", i, "bytes"), 512)));
        auto oa = on.Install(nodes.Get(si));
        oa.Start(Seconds(C.D(K("flow", i, "start"), 1.0)));
        oa.Stop(Seconds(C.D(K("flow", i, "stop"), dur)));
    }

    // ---------------- jammers ---------------- //
    const int NJ = C.I("n_jammers", 0);
    NodeContainer jn;
    if (NJ > 0)
    {
        jn.Create(NJ);
        MobilityHelper jm;
        Ptr<ListPositionAllocator> ja = CreateObject<ListPositionAllocator>();
        for (int i = 0; i < NJ; ++i)
        {
            ja->Add(Vector(C.D(K("jam", i, "x")), C.D(K("jam", i, "y")), C.D(K("jam", i, "z"))));
        }
        jm.SetPositionAllocator(ja);
        jm.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        jm.Install(jn);
        for (int i = 0; i < NJ; ++i)
        {
            JammerSpec j;
            j.id = C.S(K("jam", i, "id"));
            j.type = C.S(K("jam", i, "type"));
            j.eirp = C.D(K("jam", i, "eirp"), 15);
            j.duty = C.D(K("jam", i, "duty"), 1);
            j.dwellMs = C.D(K("jam", i, "dwell_ms"), 200);
            j.delayUs = C.D(K("jam", i, "delay_us"), 10);
            j.burstMs = C.D(K("jam", i, "burst_ms"), 1.5);
            j.pFire = C.D(K("jam", i, "p_fire"), 1);
            for (int k = 0, nc = C.I(K("jam", i, "n_ch"), 0); k < nc; ++k)
            {
                std::ostringstream kk;
                kk << "jam." << i << ".ch." << k;
                j.channels.push_back(C.I(kk.str(), 6));
            }
            WaveformGeneratorHelper wh;
            wh.SetChannel(sc);
            wh.SetTxPowerSpectralDensity(g_psd.CreateTxPowerSpectralDensity(
                std::pow(10.0, (j.eirp - 30) / 10.0),
                static_cast<uint8_t>(j.channels.empty() ? 6 : j.channels[0])));
            if (j.type == "reactive")
            {
                /* AUDIT F3a(c): WaveformGenerator::Stop() cannot truncate a wave that
                 * is already radiating -- GenerateWaveform() always emits for exactly
                 * period * dutyCycle and only the NEXT wave can be cancelled. With the
                 * default 10 ms period and duty 1.0 every trigger therefore radiated
                 * 10 ms instead of the configured burst_ms (1.5 ms). Constrain
                 * period * duty == burst_ms. Duty is held at 0.5 (rather than 1.0 with
                 * period == burst_ms) so the Stop() scheduled at delay_us + burst_ms
                 * lands inside the idle half of the period and cleanly cancels the next
                 * wave, instead of racing a GenerateWaveform() at the same instant. */
                const double burstMs = std::max(0.05, j.burstMs);
                wh.SetPhyAttribute(
                    "Period",
                    TimeValue(NanoSeconds(static_cast<int64_t>(burstMs * 2.0 * 1e6))));
                wh.SetPhyAttribute("DutyCycle", DoubleValue(0.5));
            }
            else
            {
                wh.SetPhyAttribute("Period", TimeValue(MilliSeconds(10)));
                wh.SetPhyAttribute("DutyCycle", DoubleValue(j.duty));
            }
            NetDeviceContainer jd = wh.Install(jn.Get(i));
            j.wg = jd.Get(0)
                       ->GetObject<NonCommunicatingNetDevice>()
                       ->GetPhy()
                       ->GetObject<WaveformGenerator>();
            g_jam.push_back(j);
        }
    }

    // ---------------- events ---------------- //
    for (int i = 0, ne = C.I("n_events", 0); i < ne; ++i)
    {
        double t = C.D(K("ev", i, "t"));
        std::string ty = C.S(K("ev", i, "type")), tg = C.S(K("ev", i, "target"));
        if (ty == "jammer_on" || ty == "jammer_off")
        {
            for (size_t k = 0; k < g_jam.size(); ++k)
            {
                if (g_jam[k].id != tg)
                {
                    continue;
                }
                if (ty == "jammer_on")
                {
                    if (g_jam[k].type == "reactive")
                    {
                        /* AUDIT F3a(b): this used to run `g_jam[k].on = true;` right
                         * here, during main() setup and therefore BEFORE
                         * Simulator::Run() -- the jammer was armed from t=0 and there
                         * was no clean pre-onset baseline despite truth.onset_t. A
                         * second, redundant arming via a scheduled lambda followed a
                         * few lines below; exactly one arming remains, at the
                         * scheduled onset. Armed only -- a reactive jammer radiates
                         * from ReactiveTrigger(), not from JamOn(). */
                        Simulator::Schedule(Seconds(t), &JamArmReactive, k);
                    }
                    else
                    {
                        Simulator::Schedule(Seconds(t), &JamOn, k);
                    }
                    if (g_jam[k].type == "sweep")
                    {
                        Simulator::Schedule(Seconds(t) + MilliSeconds(g_jam[k].dwellMs), &Sweep, k);
                    }
                }
                else
                {
                    Simulator::Schedule(Seconds(t), &JamOff, k);
                }
            }
        }
        else if (ty == "fade_enter")
        {
            // Flying into shadow: extra path loss AND loss of the LOS component, so
            // the channel degrades from Rician-like (m>1) to Rayleigh (m=1).
            double depth = C.D(K("ev", i, "depth_db"), 15.0);
            double newm = C.D(K("ev", i, "nakagami_m"), 1.0);
            double base = C.D("ref_loss", 40.0);
            Ptr<LogDistancePropagationLossModel> plc = pl;
            Ptr<NakagamiPropagationLossModel> nkc = g_nak;
            Simulator::Schedule(Seconds(t), [plc, nkc, base, depth, newm]() {
                plc->SetAttribute("ReferenceLoss", DoubleValue(base + depth));
                if (nkc && newm > 0)
                {
                    nkc->SetAttribute("m0", DoubleValue(newm));
                    nkc->SetAttribute("m1", DoubleValue(newm));
                    nkc->SetAttribute("m2", DoubleValue(newm));
                }
            });
        }
        else if (ty == "fade_exit")
        {
            double base = C.D("ref_loss", 40.0);
            Ptr<LogDistancePropagationLossModel> plc = pl;
            Simulator::Schedule(Seconds(t), [plc, base]() {
                plc->SetAttribute("ReferenceLoss", DoubleValue(base));
            });
        }
        else if (ty == "node_down")
        {
            for (int k = 0; k < N; ++k)
            {
                if (ids[k] == tg)
                {
                    Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devs.Get(k));
                    Simulator::Schedule(Seconds(t), &WifiPhy::SetOffMode, wd->GetPhy());
                    Ptr<MeshNodeApp> ap = apps[k];
                    Simulator::Schedule(Seconds(t), &MeshNodeApp::SetTxEnabled, ap, false);
                }
            }
        }
    }

    // reactive jammers hook the agent's transmit start
    {
        Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devs.Get(g_agentIdx));
        wd->GetPhy()->TraceConnectWithoutContext("PhyTxBegin", MakeCallback(&ReactiveTrigger));
        wd->GetPhy()->TraceConnectWithoutContext("PhyTxBegin", MakeCallback(&AgentTxBegin));
        wd->GetPhy()->TraceConnectWithoutContext("PhyTxEnd", MakeCallback(&AgentTxEnd));
        wd->GetPhy()->TraceConnectWithoutContext("PhyTxBegin", MakeCallback(&AgentTxBeginKpi));
        wd->GetPhy()->TraceConnectWithoutContext("PhyTxEnd", MakeCallback(&AgentTxEndKpi));
        wd->GetRemoteStationManager()->TraceConnectWithoutContext(
            "MacTxFinalDataFailed", MakeCallback(&MacFinalFail));
        wd->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx", MakeCallback(&SnifferRx));
        wd->GetPhy()->TraceConnectWithoutContext("PhyRxDrop", MakeCallback(&PhyRxDrop));
    }

    // spectrum analyzer beside the agent
    {
        NodeContainer an;
        an.Create(1);
        MobilityHelper am;
        Ptr<ListPositionAllocator> aa = CreateObject<ListPositionAllocator>();
        aa->Add(Vector(C.D(K("node", g_agentIdx, "x")),
                       C.D(K("node", g_agentIdx, "y")),
                       C.D(K("node", g_agentIdx, "z"))));
        am.SetPositionAllocator(aa);
        am.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        am.Install(an);
        SpectrumAnalyzerHelper sah;
        sah.SetChannel(sc);
        Ptr<SpectrumValue> probe = g_psd.CreateConstant(0.0);
        sah.SetRxSpectrumModel(ConstCast<SpectrumModel>(probe->GetSpectrumModel()));
        sah.SetPhyAttribute("Resolution", TimeValue(MicroSeconds(500)));
        sah.SetPhyAttribute("NoisePowerSpectralDensity", DoubleValue(4.14e-21));
        NetDeviceContainer ad = sah.Install(an);
        // AUDIT F3g: track the agent rather than sitting at its t=0 coordinates.
        g_agentMob = nodes.Get(g_agentIdx)->GetObject<MobilityModel>();
        g_analyzerMob = an.Get(0)->GetObject<MobilityModel>();
        Simulator::ScheduleNow(&SyncAnalyzerPosLoop);
        Ptr<SpectrumAnalyzer> sa =
            ad.Get(0)->GetObject<NonCommunicatingNetDevice>()->GetPhy()->GetObject<SpectrumAnalyzer>();
        sa->TraceConnectWithoutContext("AveragePowerSpectralDensityReport",
                                       MakeCallback(&AnalyzerReport));
        sa->Start();
    }

    // ---------------- sampling ---------------- //
    std::ofstream pf(out + ".percept.csv"), tf(out + ".truth.csv"), rf(out + ".routes.csv");
    pf << "t,channel,pdr_mean,pdr_worst,pdr_spread,n_links,links_degraded,rssi_mean,"
          "rssi_per_link,pdr_per_link,noise_dbm,meas_floor_dbm,band_power,hb_gap_max,"
          "data_tx,data_rx,rx_ok,rx_err,"
          "rev_rssi_per_link,rev_pdr_per_link,rev_age_per_link,"
          "tx_attempts,tx_acked,tx_defer_ms,tx_retry_depth,"
          "shadow_exp,shadow_miss,silent_exp,silent_miss\n";
    tf << "t,cause,jammer_on,jammer_channels\n";
    rf << "t,node,n_routes,max_hops,reaches_dst\n";

    const std::string cause = C.S("truth_cause", "unknown");
    const double beaconHz = 10.0;
    static double lastSample = 0.0;
    static uint32_t lastDataRx = 0;
    // AUDIT F3e: per-peer sliding window of (sample time, beacons received in that sample),
    // so the offline CSV reports the same 1 s per-link PDR the live bridge does.
    constexpr double PDR_WIN_S = 1.0;
    static std::map<int, std::deque<std::pair<double, uint32_t>>> s_pdrHist;

    std::function<void()> tick = [&]() {
        double t = Simulator::Now().GetSeconds();
        double win = t - lastSample;
        lastSample = t;
        Ptr<MeshNodeApp> me = apps[g_agentIdx];

        // ---- true per-link PDR from sequence-counted beacons ----
        std::ostringstream rssiS, pdrS;
        std::vector<double> pdrs;
        double rssiSum = 0;
        int rssiN = 0, degraded = 0;
        double hbMax = 0;
        const auto& rc = me->RxCounts();
        const auto& lh = me->LastHeard();
        int nl = 0;
        for (int i = 0; i < N; ++i)
        {
            if ((uint32_t)i == g_agentIdx)
            {
                continue;
            }
            auto it = rc.find(i);
            uint32_t got = (it == rc.end()) ? 0 : it->second;
            /* AUDIT F3e: with a 100 ms sample window and 10 Hz beacons, expect == 1, so
             * this ratio quantised to {0, 1} -- every per-link PDR in the offline corpus
             * was binary while the LIVE bridge used a 1 s sliding window and produced a
             * real fraction. Training and inference therefore saw different distributions
             * for the most basic feature in the system. Accumulate over the same 1 s
             * window the bridge uses so the two agree. */
            s_pdrHist[i].push_back({t, got});
            while (s_pdrHist[i].size() > 1 && t - s_pdrHist[i].front().first > PDR_WIN_S)
            {
                s_pdrHist[i].pop_front();
            }
            uint32_t gotWin = 0;
            for (const auto& e : s_pdrHist[i])
            {
                gotWin += e.second;
            }
            double span = std::max(0.3, t - s_pdrHist[i].front().first + win);
            double expect = std::max(1.0, beaconHz * span);
            double pdr = std::min(1.0, gotWin / expect);
            // only count peers that were ever in range as links
            auto lit = lh.find(i);
            double age = (lit == lh.end()) ? 99.0 : (t - lit->second);
            if (lit == lh.end() && got == 0 && t < 8.0)
            {
                continue; // never heard, still warming up
            }
            if (lit == lh.end() && got == 0)
            {
                continue; // out of range: not our neighbour
            }
            nl++;
            pdrs.push_back(pdr);
            if (pdr < 0.6)
            {
                degraded++;
            }
            hbMax = std::max(hbMax, age);
            auto rit = g_linkRssi.find(i);
            double r = (rit == g_linkRssi.end()) ? -120.0 : rit->second;
            if (rit != g_linkRssi.end())
            {
                rssiSum += r;
                rssiN++;
            }
            rssiS << (nl > 1 ? " " : "") << std::fixed << std::setprecision(1) << r;
            pdrS << (nl > 1 ? " " : "") << std::fixed << std::setprecision(3) << pdr;
        }
        me->ResetWindow();

        double pm = 0, pw = 1, ps = 0;
        if (!pdrs.empty())
        {
            for (double x : pdrs)
            {
                pm += x;
                pw = std::min(pw, x);
            }
            pm /= pdrs.size();
            for (double x : pdrs)
            {
                ps += (x - pm) * (x - pm);
            }
            ps = std::sqrt(ps / pdrs.size());
        }
        else
        {
            pw = 0;
        }

        const std::vector<double>& fv = (g_bandFloor.size() == g_nCh) ? g_bandFloor : g_band;
        double meas = (g_curCh >= 1 && g_curCh - 1 < (int)fv.size() && fv[g_curCh - 1] < 1e8) ? fv[g_curCh - 1] : -200;
        // band_power is now ENERGY (mean over agent-silent reports), not the min-hold:
        // a duty-cycled jammer is invisible to a minimum. meas_floor_dbm above keeps the
        // min-hold, so the two questions have two answers.
        const std::vector<double> be = BandEnergy(g_bandSamp, fv);
        std::ostringstream bp;
        for (size_t c = 0; c < be.size(); ++c)
        {
            bp << (c ? " " : "") << std::fixed << std::setprecision(1) << be[c];
        }
        g_bandFloor.assign(g_nCh, 1e9);
        for (auto& v : g_bandSamp) { v.clear(); }

        // ---- reciprocal reports + TX KPIs + TX-shadow (TELEMETRY.md) ----
        std::ostringstream revR, revP, revA;
        const auto& rev = me->Reverse();
        bool rfirst = true;
        for (int i = 0; i < N; ++i)
        {
            if ((uint32_t)i == g_agentIdx)
            {
                continue;
            }
            auto lit = lh.find(i);
            if (lit == lh.end())
            {
                continue;
            }
            auto rit = rev.find(i);
            double rr = -120.0, rp = -1.0, ra = 999.0;
            if (rit != rev.end())
            {
                rr = std::get<0>(rit->second);
                rp = std::get<1>(rit->second);
                ra = t - std::get<2>(rit->second);
            }
            if (!rfirst)
            {
                revR << " ";
                revP << " ";
                revA << " ";
            }
            revR << std::setprecision(1) << rr;
            revP << std::setprecision(3) << rp;
            revA << std::setprecision(2) << ra;
            rfirst = false;
        }
        static uint64_t sAtt = 0, sFail = 0;
        static double sDefer = 0.0;
        static uint32_t sDeferN = 0;
        uint64_t dAtt = g_txAttempts - sAtt;
        uint64_t dFail = g_txFinalFail - sFail;
        sAtt = g_txAttempts;
        sFail = g_txFinalFail;
        double dDefer = g_deferAccum - sDefer;
        uint32_t dDeferN = g_deferN - sDeferN;
        sDefer = g_deferAccum;
        sDeferN = g_deferN;
        double deferMs = dDeferN ? dDefer / dDeferN : 0.0;
        uint64_t dAck = (dAtt > dFail) ? (dAtt - dFail) : 0;
        // AUDIT F3b: shared read+reset, so `decide` (same instant) sees the same data.
        const ShadowSample& sh = SampleShadow(me);
        uint32_t shE = sh.shE, shM = sh.shM;
        uint32_t siE = sh.siE, siM = sh.siM;

        uint32_t drx = me->DataRxTotal();
        pf << std::fixed << std::setprecision(3) << t << "," << g_curCh << "," << pm << "," << pw << ","
           << ps << "," << nl << "," << degraded << ","
           << (rssiN ? rssiSum / rssiN : -120.0) << ",\"" << rssiS.str() << "\",\"" << pdrS.str()
           << "\"," << g_lastNoise << "," << meas << ",\"" << bp.str() << "\"," << hbMax << ","
           /* AUDIT: was apps[0]->DataSent() regardless of which node the flow source is,
            * so data_tx read 0 for every scenario whose source is not node 0 -- including
            * hidden_terminal, where it silently disabled the flow.0.start check. */
           << apps[g_flowSrcIdx]->DataSent() << "," << (drx - lastDataRx) << "," << g_rxOk
           << "," << g_rxErr
           << ",\"" << revR.str() << "\",\"" << revP.str() << "\",\"" << revA.str() << "\","
           << dAtt << "," << dAck << "," << std::setprecision(2) << deferMs << ","
           << (dAtt ? double(dFail) * 3.0 / dAtt : 0.0) << ","
           << shE << "," << shM << "," << siE << "," << siM << "\n";
        lastDataRx = drx;

        bool on = false;
        std::ostringstream chs;
        for (auto& j : g_jam)
        {
            if (j.on)
            {
                on = true;
                if (j.type == "sweep" && !j.channels.empty())
                {
                    chs << j.channels[j.curIdx] << " ";
                }
                else
                {
                    for (int c : j.channels)
                    {
                        chs << c << " ";
                    }
                }
            }
        }
        tf << std::fixed << std::setprecision(3) << t << "," << cause << "," << (on ? 1 : 0)
           << ",\"" << chs.str() << "\"\n";

        if (t + 0.1 < dur)
        {
            Simulator::Schedule(MilliSeconds(100), tick);
        }
    };
    Simulator::Schedule(Seconds(1.0), tick);

    // ---- OLSR route-table verification, once a second ----
    std::function<void()> routeTick = [&]() {
        double t = Simulator::Now().GetSeconds();
        for (int i = 0; i < N; ++i)
        {
            Ptr<Ipv4> ip = nodes.Get(i)->GetObject<Ipv4>();
            Ptr<Ipv4RoutingProtocol> rp = ip->GetRoutingProtocol();
            Ptr<Ipv4ListRouting> lrp = DynamicCast<Ipv4ListRouting>(rp);
            int nroutes = 0, maxh = 0, reaches = 0;
            if (lrp)
            {
                for (uint32_t k = 0; k < lrp->GetNRoutingProtocols(); ++k)
                {
                    int16_t prio;
                    Ptr<Ipv4RoutingProtocol> sub = lrp->GetRoutingProtocol(k, prio);
                    Ptr<olsr::RoutingProtocol> o = DynamicCast<olsr::RoutingProtocol>(sub);
                    if (!o)
                    {
                        continue;
                    }
                    auto entries = o->GetRoutingTableEntries();
                    nroutes = (int)entries.size();
                    for (auto& e : entries)
                    {
                        maxh = std::max(maxh, (int)e.distance);
                        if (e.destAddr == ifs.GetAddress(dstIdx))
                        {
                            reaches = 1;
                        }
                    }
                }
            }
            rf << std::fixed << std::setprecision(1) << t << "," << ids[i] << "," << nroutes << ","
               << maxh << "," << reaches << "\n";
        }
        if (t + 1.0 < dur)
        {
            Simulator::Schedule(Seconds(1.0), routeTick);
        }
    };
    Simulator::Schedule(Seconds(2.0), routeTick);

    // ---------------- live agent bridge (1 Hz decision epochs) ---------------- //
    bool bridged = false;
    if (!sockPath.empty())
    {
        bridged = BridgeConnect(sockPath);
        if (!bridged)
        {
            std::cerr << "[sim] WARNING: could not connect to " << sockPath
                      << "; running offline\n";
        }
    }
    static int g_hops = 0, g_declared = 0;
    std::function<void()> decide = [&]() {
        double t = Simulator::Now().GetSeconds();
        if (g_sock < 0 || g_declared)
        {
            return;
        }
        Ptr<MeshNodeApp> me = apps[g_agentIdx];

        // ---- assemble the state the agent may see (no ground truth) ----
        std::ostringstream js;
        js << "{\"t\":" << std::fixed << std::setprecision(2) << t << ",\"channel\":" << g_curCh
           << ",\"n_channels\":" << g_nCh << ",\"pdr\":[";
        // Trailing 1 s window over cumulative counts (a 100 ms window would hold
        // ~1 beacon per peer and quantise PDR to {0,1}).
        static std::deque<std::pair<double, std::map<uint32_t, uint32_t>>> s_hist;
        const auto& rc = me->RxTotals();
        const auto& lh = me->LastHeard();
        s_hist.emplace_back(t, rc);
        while (s_hist.size() > 1 && t - s_hist.front().first > 1.0)
        {
            s_hist.pop_front();
        }
        const auto& oldCounts = s_hist.front().second;
        double win = std::max(0.3, t - s_hist.front().first);
        bool first = true;
        std::ostringstream rs, hb;
        int nl = 0;
        for (int i = 0; i < N; ++i)
        {
            if ((uint32_t)i == g_agentIdx)
            {
                continue;
            }
            auto lit = lh.find(i);
            if (lit == lh.end())
            {
                continue;
            }
            auto it = rc.find(i);
            uint32_t tot = (it == rc.end()) ? 0 : it->second;
            auto ot = oldCounts.find(i);
            uint32_t old = (ot == oldCounts.end()) ? 0 : ot->second;
            uint32_t got = (tot >= old) ? (tot - old) : 0;
            double pdr = std::min(1.0, got / (10.0 * win));
            auto rit = g_linkRssi.find(i);
            double r = (rit == g_linkRssi.end()) ? -120.0 : rit->second;
            if (!first)
            {
                js << ",";
                rs << ",";
                hb << ",";
            }
            js << std::fixed << std::setprecision(3) << pdr;
            rs << std::fixed << std::setprecision(1) << r;
            hb << std::fixed << std::setprecision(2) << (t - lit->second);
            first = false;
            nl++;
        }
        // NOTE: every ostringstream below MUST carry std::fixed. Without it,
        // setprecision(n) means n SIGNIFICANT DIGITS in the default float format,
        // so setprecision(1) emitted "-1e+02" for every noise floor between -95
        // and -104 dBm and for every RSSI near -100. The CSV emitter had
        // std::fixed; this one did not. Result: through the live bridge the whole
        // spectrum collapsed to a constant -100 on all 8 channels, so S2
        // (noise_delta_base), S5 (scan_noise_spread) and noise_std were
        // identically zero for entire episodes -- the two decisive discrimination
        // statistics in the design, dead, silently. The LLM teacher scored 30.8%
        // on ns-3 vs 65.6% on refsim and the student 18.8% vs 85% on the CSV
        // corpus, all from this one missing manipulator.
        const std::vector<double>& fv =
            (g_bandFloorBridge.size() == g_nCh) ? g_bandFloorBridge : g_band;
        const std::vector<double> be = BandEnergy(g_bandSampBridge, fv);
        std::ostringstream bp;
        for (size_t c = 0; c < be.size(); ++c)
        {
            bp << (c ? "," : "") << std::fixed << std::setprecision(1) << be[c];
        }
        g_bandFloorBridge.assign(g_nCh, 1e9);   // bridge resets its OWN accumulators
        for (auto& v : g_bandSampBridge) { v.clear(); }
        static uint64_t s_rxOk = 0, s_rxErr = 0;
        static uint32_t s_dsent = 0;
        uint64_t dOk = g_rxOk - s_rxOk, dErr = g_rxErr - s_rxErr;
        s_rxOk = g_rxOk;
        s_rxErr = g_rxErr;
        double retry = (dOk + dErr) ? double(dErr) / double(dOk + dErr) : 0.0;
        uint32_t dsent = apps[g_agentIdx]->DataSent();
        double sentRate = (dsent - s_dsent) / std::max(0.2, win);
        s_dsent = dsent;
        double nominal = std::max(1.0, fkbps * 1000.0 / (8.0 * fbytes));
        double load = std::min(1.0, sentRate / nominal);
        // --- telemetry for the live agent (TELEMETRY.md) ---
        std::ostringstream jrevR, jrevP, jrevA;
        const auto& jrev = me->Reverse();
        bool jf = true;
        for (int i = 0; i < N; ++i)
        {
            if ((uint32_t)i == g_agentIdx || lh.find(i) == lh.end())
            {
                continue;
            }
            auto rit = jrev.find(i);
            double rr = -120.0, rp = -1.0, ra = 999.0;
            if (rit != jrev.end())
            {
                rr = std::get<0>(rit->second);
                rp = std::get<1>(rit->second);
                ra = t - std::get<2>(rit->second);
            }
            if (!jf) { jrevR << ","; jrevP << ","; jrevA << ","; }
            jrevR << std::fixed << std::setprecision(1) << rr;
            jrevP << std::fixed << std::setprecision(3) << rp;
            jrevA << std::fixed << std::setprecision(2) << ra;
            jf = false;
        }
        static uint64_t bAtt = 0, bFail = 0;
        static double bDefer = 0.0;
        static uint32_t bDeferN = 0;
        uint64_t bdA = g_txAttempts - bAtt, bdF = g_txFinalFail - bFail;
        bAtt = g_txAttempts; bFail = g_txFinalFail;
        double bdD = g_deferAccum - bDefer;
        uint32_t bdN = g_deferN - bDeferN;
        bDefer = g_deferAccum; bDeferN = g_deferN;
        // AUDIT F3b: shared read+reset (see SampleShadow) -- `tick` runs at this same
        // simulation time and used to consume and clear these counters first.
        const ShadowSample& bsh = SampleShadow(me);
        js << "],\"rev_rssi\":[" << jrevR.str() << "],\"rev_pdr\":[" << jrevP.str()
           << "],\"rev_age\":[" << jrevA.str() << "],\"tx_att\":" << bdA
           << ",\"tx_ack\":" << (bdA > bdF ? bdA - bdF : 0)
           << ",\"tx_defer_ms\":" << std::setprecision(3) << (bdN ? bdD / bdN : 0.0)
           << ",\"sh_e\":" << bsh.shE << ",\"sh_m\":" << bsh.shM
           << ",\"si_e\":" << bsh.siE << ",\"si_m\":" << bsh.siM
           << ",\"rssi\":[" << rs.str() << "],\"hb\":[" << hb.str() << "],\"band\":["
           << bp.str() << "],\"n_links\":" << nl << ",\"hops_used\":" << g_hops
           << ",\"tx_power\":" << curTxp << ",\"retry\":" << std::setprecision(3) << retry
           << ",\"load\":" << load << ",\"tx_duty\":" << (g_agentTx ? 1.0 : 0.35)
           // ROBUSTNESS (capstone): raw MAC-layer decoded-frame rate in this window
           // (dOk already computed above for `retry`). SnifferRx fires on ANY
           // successfully-decoded 802.11 frame the agent's PHY hears, mesh peer or
           // not, unlike pdr_per_link which is mesh-link-only by construction -- this
           // is the signal foreign_fps was supposed to carry and previously did not.
           // The Python adapter rescales it to match refsim's decodable_fps range.
           << ",\"decod_fps\":" << std::setprecision(1) << (win > 0 ? dOk / win : 0.0) << "}";

        if (!BridgeSend(js.str()))
        {
            g_sock = -1;
            return;
        }
        std::string reply;
        if (!BridgeRecv(reply))
        {
            g_sock = -1;
            return;
        }

        // ---- apply the action, charging its real cost ----
        std::string call = JGet(reply, "call");
        if (call == "hop_channel")
        {
            int ch = (int)JNum(reply, "channel", g_curCh);
            ch = std::max(1, std::min((int)g_nCh, ch));
            // A mesh-wide coordinated hop: every node retunes. The cost is the
            // resynchronisation gap, which shows up as lost beacons.
            for (int i = 0; i < N; ++i)
            {
                Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devs.Get(i));
                std::ostringstream cs;
                cs << "{" << ch << ", 20, BAND_2_4GHZ, 0}";
                wd->GetPhy()->SetAttribute("ChannelSettings", StringValue(cs.str()));
            }
            g_curCh = ch;
            g_hops++;
        }
        else if (call == "set_tx_power")
        {
            double dbm = JNum(reply, "dbm", 16.0);
            Ptr<WifiNetDevice> wd = DynamicCast<WifiNetDevice>(devs.Get(g_agentIdx));
            wd->GetPhy()->SetAttribute("TxPowerStart", DoubleValue(dbm));
            wd->GetPhy()->SetAttribute("TxPowerEnd", DoubleValue(dbm));
            // AUDIT: keep the reported tx_power in step with the radio, otherwise the
            // agent can never observe the effect of its own set_tx_power.
            curTxp = dbm;
        }
        else if (call == "change_tdma_slot")
        {
            apps[g_agentIdx]->SetSlot((uint8_t)JNum(reply, "slot", 1));
        }
        else if (call == "silent_listen")
        {
            double ms = JNum(reply, "duration_ms", 200.0);
            apps[g_agentIdx]->SetTxEnabled(false);
            Ptr<MeshNodeApp> a = apps[g_agentIdx];
            Simulator::Schedule(MilliSeconds(ms), &MeshNodeApp::SetTxEnabled, a, true);
        }
        else if (call == "move")
        {
            auto cv = nodes.Get(g_agentIdx)->GetObject<ConstantVelocityMobilityModel>();
            Vector p0 = cv->GetPosition();
            cv->SetPosition(Vector(p0.x + JNum(reply, "dx"), p0.y + JNum(reply, "dy"),
                                   p0.z + JNum(reply, "dz")));
            SyncAnalyzerPos();   // AUDIT F3g: move the analyzer with the agent
        }
        else if (call == "declare_link_lost")
        {
            g_declared = 1;
            std::cout << "[sim] agent declared link lost at t=" << t << "\n";
        }
        if (t + 0.1 < dur)
        {
            Simulator::Schedule(MilliSeconds(100), decide);
        }
    };
    if (bridged)
    {
        Simulator::Schedule(Seconds(1.0), decide);
    }

    Simulator::Stop(Seconds(dur));
    std::cout << "[sim] " << C.S("name") << " nodes=" << N << " jammers=" << NJ
              << " truth=" << cause << " tdmaSlots=" << tdmaSlots << "\n";
    Simulator::Run();
    Simulator::Destroy();
    pf.close();
    tf.close();
    rf.close();
    std::cout << "[sim] done rxOk=" << g_rxOk << " rxErr=" << g_rxErr
              << " analyzerClean=" << g_cleanReports << " analyzerDirty=" << g_dirtyReports
              << "\n";
    for (int i = 0; i < N; ++i)
    {
        std::cout << "[diag] node " << ids[i] << " beaconsSent=" << apps[i]->BeaconsSent()
                  << " dataSent=" << apps[i]->DataSent() << " dataRx=" << apps[i]->DataRxTotal()
                  << " peersHeard=" << apps[i]->LastHeard().size() << "\n";
    }
    return 0;
}
