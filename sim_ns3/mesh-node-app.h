/*
 * MeshNodeApp — the application/measurement plane for the jamming testbed.
 *
 * Replaces OnOffApplication so that we get, exactly and per-link:
 *   - application-level delivery ratio (sequence-numbered, not a PHY proxy)
 *   - per-peer heartbeat liveness (the S9 statistic)
 *   - real TDMA slotting on the control plane (so change_tdma_slot() means something)
 *
 * Every node runs one. It emits:
 *   BEACON  every beaconInterval  -> broadcast, how peers measure this node's link
 *   DATA    at dataRateKbps       -> unicast to a destination, the mission traffic
 *
 * Each packet carries {srcIdx, seq, kind}. A receiver counts arrivals per source in
 * a sliding window and divides by what should have arrived -> true per-link PDR.
 */
#ifndef MESH_NODE_APP_H
#define MESH_NODE_APP_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/inet-socket-address.h"
#include "ns3/ipv4-address.h"
#include "ns3/nstime.h"
#include "ns3/socket.h"
#include "ns3/traced-callback.h"

#include <functional>
#include <map>
#include <tuple>
#include <vector>

namespace ns3
{

#pragma pack(push, 1)
struct MeshPktHdr
{
    uint32_t srcIdx;
    uint32_t seq;
    uint8_t kind;      // 0 = beacon, 1 = data
    uint8_t slot;
    uint8_t nReports;  // reciprocal link reports that follow (TELEMETRY.md §2)
};

/** "This is how I hear YOU" -- echoed in every beacon so neighbours learn the
 *  REVERSE link. 3 bytes each; an ESP-NOW/painlessMesh beacon carries these just as
 *  easily as an ns-3 one. */
struct LinkReport
{
    uint8_t peerIdx;
    int8_t rssiDbm;    // clamped to [-128, 0]
    uint8_t pdrQ;      // delivery ratio quantised to 0..255
};
#pragma pack(pop)

/** Set by the simulation so the app can ask "were we transmitting at time t?" --
 *  the only input the TX-shadow statistic needs beyond beacon periodicity. */
using TxShadowQuery = std::function<bool(double)>;

class MeshNodeApp : public Application
{
  public:
    static TypeId GetTypeId();
    MeshNodeApp();
    ~MeshNodeApp() override;

    void Setup(uint32_t myIdx,
               Ipv4Address bcast,
               uint16_t port,
               Time beaconInterval,
               double dataKbps,
               uint32_t pktBytes,
               Ipv4Address dataDst,
               bool hasData);

    /** TDMA: only transmit during our slot. slots<=1 disables slotting (pure CSMA). */
    void SetTdma(uint8_t slot, uint8_t nSlots, Time frame);
    uint8_t GetSlot() const { return m_slot; }
    void SetSlot(uint8_t s) { m_slot = s; }
    void SetTxEnabled(bool e) { m_txEnabled = e; }
    void SetDataRate(double kbps) { m_dataKbps = kbps; }

    /** AUDIT F3d: active window of the mission DATA stream (flow.0.start /
     *  flow.0.stop in the scenario config). Beacons are deliberately not affected --
     *  they are the measurement plane and must run for the whole episode. Must be
     *  called before the application starts. Without it the data stream behaves as
     *  before: it begins ~50 ms after StartApplication() and runs to StopTime. */
    void SetDataWindow(Time start, Time stop);

    /** Per-source receive counters (index -> count), reset by the sampler. */
    const std::map<uint32_t, uint32_t>& RxCounts() const { return m_rxCount; }
    /** Cumulative, never reset -- each consumer diffs against its own snapshot. */
    const std::map<uint32_t, uint32_t>& RxTotals() const { return m_rxTotal; }
    const std::map<uint32_t, double>& LastHeard() const { return m_lastHeard; }
    uint32_t BeaconsSent() const { return m_beaconSeq; }
    uint32_t DataSent() const { return m_dataSeq; }
    uint32_t DataRxTotal() const { return m_dataRx; }
    void ResetWindow() { m_rxCount.clear(); }

    /** Per-peer RSSI this node measured at the PHY (filled by the sim's sniffer). */
    std::map<uint32_t, double>& RxRssi() { return m_rxRssi; }
    /** What peers told US about how well they hear us: peer -> (rssi, pdr, t). */
    const std::map<uint32_t, std::tuple<double, double, double>>& Reverse() const
    {
        return m_reverse;
    }
    void SetTxShadowQuery(TxShadowQuery q) { m_inTxShadow = q; }
    /** Called immediately before each transmission, so the sim can timestamp the
     *  enqueue and later measure enqueue -> on-air delay (tx_defer). */
    void SetOnSend(std::function<void()> cb) { m_onSend = cb; }
    /** TX-shadow buckets (TELEMETRY.md §1). */
    uint32_t ShadowExpected() const { return m_shadowExp; }
    uint32_t ShadowMissed() const { return m_shadowMiss; }
    uint32_t SilentExpected() const { return m_silentExp; }
    uint32_t SilentMissed() const { return m_silentMiss; }
    void ResetShadow() { m_shadowExp = m_shadowMiss = m_silentExp = m_silentMiss = 0; }

    /** True when the current time falls in this node's TDMA slot. */
    bool InMySlot() const;
    /** Delay until the start of this node's next slot (or the plain beacon
     *  interval when slotting is disabled). A TDMA MAC must transmit AT its slot
     *  boundary; scheduling on a free-running timer means most beacons land in
     *  someone else's slot and are suppressed. */
    Time TimeToMySlot() const;

  private:
    void StartApplication() override;
    void StopApplication() override;
    void SendBeacon();
    void SendData();
    void StopData();
    void HandleRead(Ptr<Socket> s);

    Ptr<Socket> m_txSock;
    Ptr<Socket> m_rxSock;
    uint32_t m_myIdx{0};
    Ipv4Address m_bcast;
    Ipv4Address m_dataDst;
    uint16_t m_port{9999};
    Time m_beaconInterval{MilliSeconds(100)};
    double m_dataKbps{0.0};
    uint32_t m_pktBytes{512};
    bool m_hasData{false};
    bool m_txEnabled{true};

    uint8_t m_slot{0};
    uint8_t m_nSlots{0}; // 0/1 == no TDMA
    Time m_frame{MilliSeconds(100)};

    uint32_t m_beaconSeq{0};
    uint32_t m_dataSeq{0};
    uint32_t m_dataRx{0};
    EventId m_beaconEv;
    EventId m_dataEv;
    EventId m_dataStopEv;
    // AUDIT F3d: mission-flow window; m_hasDataWindow == false keeps the old behaviour
    Time m_dataStart{Seconds(0)};
    Time m_dataStop{Seconds(0)};
    bool m_hasDataWindow{false};
    std::map<uint32_t, uint32_t> m_rxCount;
    std::map<uint32_t, uint32_t> m_rxTotal;
    std::map<uint32_t, double> m_rxRssi;                 // PHY-measured, per peer
    std::map<uint32_t, double> m_rxPdr;                  // our view of each peer
    std::map<uint32_t, std::tuple<double, double, double>> m_reverse;
    std::map<uint32_t, uint32_t> m_lastSeq;              // for gap-based miss detection
    std::map<uint32_t, double> m_lastSeqTime;
    TxShadowQuery m_inTxShadow;
    std::function<void()> m_onSend;
    std::map<uint32_t, uint32_t> m_expTotal;   // beacons that SHOULD have arrived
    std::map<uint32_t, uint32_t> m_rcvTotal;   // beacons that did
    uint32_t m_shadowExp{0}, m_shadowMiss{0}, m_silentExp{0}, m_silentMiss{0};
    std::map<uint32_t, double> m_lastHeard;
};

} // namespace ns3
#endif
