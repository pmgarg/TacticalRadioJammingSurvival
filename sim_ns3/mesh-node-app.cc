#include "mesh-node-app.h"

#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/uinteger.h"

#include <algorithm>
#include <cstring>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("MeshNodeApp");
NS_OBJECT_ENSURE_REGISTERED(MeshNodeApp);

TypeId
MeshNodeApp::GetTypeId()
{
    static TypeId tid = TypeId("ns3::MeshNodeApp")
                            .SetParent<Application>()
                            .SetGroupName("Applications")
                            .AddConstructor<MeshNodeApp>();
    return tid;
}

MeshNodeApp::MeshNodeApp() = default;
MeshNodeApp::~MeshNodeApp() = default;

void
MeshNodeApp::Setup(uint32_t myIdx,
                   Ipv4Address bcast,
                   uint16_t port,
                   Time beaconInterval,
                   double dataKbps,
                   uint32_t pktBytes,
                   Ipv4Address dataDst,
                   bool hasData)
{
    m_myIdx = myIdx;
    m_bcast = bcast;
    m_port = port;
    m_beaconInterval = beaconInterval;
    m_dataKbps = dataKbps;
    m_pktBytes = pktBytes;
    m_dataDst = dataDst;
    m_hasData = hasData;
}

void
MeshNodeApp::SetTdma(uint8_t slot, uint8_t nSlots, Time frame)
{
    m_slot = slot;
    m_nSlots = nSlots;
    m_frame = frame;
}

bool
MeshNodeApp::InMySlot() const
{
    if (m_nSlots <= 1)
    {
        return true; // slotting disabled -> plain CSMA
    }
    int64_t now = Simulator::Now().GetNanoSeconds();
    int64_t frame = m_frame.GetNanoSeconds();
    int64_t slotLen = frame / m_nSlots;
    int64_t within = now % frame;
    int64_t idx = (slotLen > 0) ? (within / slotLen) : 0;
    return idx == static_cast<int64_t>(m_slot % m_nSlots);
}

Time
MeshNodeApp::TimeToMySlot() const
{
    if (m_nSlots <= 1)
    {
        return m_beaconInterval;
    }
    int64_t now = Simulator::Now().GetNanoSeconds();
    int64_t frame = m_frame.GetNanoSeconds();
    int64_t slotLen = frame / m_nSlots;
    int64_t mySlotStart = slotLen * (m_slot % m_nSlots);
    int64_t within = now % frame;
    int64_t delta = mySlotStart - within;
    if (delta <= 0)
    {
        delta += frame;
    }
    return NanoSeconds(delta);
}

void
MeshNodeApp::StartApplication()
{
    TypeId tid = UdpSocketFactory::GetTypeId();
    m_rxSock = Socket::CreateSocket(GetNode(), tid);
    m_rxSock->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
    m_rxSock->SetRecvCallback(MakeCallback(&MeshNodeApp::HandleRead, this));

    m_txSock = Socket::CreateSocket(GetNode(), tid);
    m_txSock->SetAllowBroadcast(true);
    m_txSock->Bind();

    // Under TDMA, align to our slot. Without TDMA, stagger to avoid a thundering herd.
    Time first = (m_nSlots > 1) ? TimeToMySlot() : MilliSeconds(m_myIdx * 7 % 100);
    m_beaconEv = Simulator::Schedule(first, &MeshNodeApp::SendBeacon, this);
    if (m_hasData && m_dataKbps > 0)
    {
        m_dataEv = Simulator::Schedule(MilliSeconds(50), &MeshNodeApp::SendData, this);
    }
}

void
MeshNodeApp::StopApplication()
{
    Simulator::Cancel(m_beaconEv);
    Simulator::Cancel(m_dataEv);
    if (m_rxSock)
    {
        m_rxSock->Close();
    }
    if (m_txSock)
    {
        m_txSock->Close();
    }
}

void
MeshNodeApp::SendBeacon()
{
    if (m_txEnabled && InMySlot())
    {
        // Attach what we measured at the PHY for each neighbour we hear. A receiver
        // pulls out the entry about ITSELF and thereby learns the reverse link -- how
        // well it is being heard. (TELEMETRY.md §2)
        uint8_t buf[128] = {0};
        MeshPktHdr h{m_myIdx, m_beaconSeq, 0, m_slot, 0};
        size_t off = sizeof(MeshPktHdr);
        for (auto& kv : m_rxRssi)
        {
            if (off + sizeof(LinkReport) > sizeof(buf) || h.nReports >= 16)
            {
                break;
            }
            double pdr = 0.0;
            auto it = m_rxPdr.find(kv.first);
            if (it != m_rxPdr.end())
            {
                pdr = it->second;
            }
            LinkReport r;
            r.peerIdx = static_cast<uint8_t>(kv.first);
            r.rssiDbm = static_cast<int8_t>(std::max(-128.0, std::min(0.0, kv.second)));
            r.pdrQ = static_cast<uint8_t>(std::max(0.0, std::min(1.0, pdr)) * 255.0);
            std::memcpy(buf + off, &r, sizeof(r));
            off += sizeof(r);
            h.nReports++;
        }
        std::memcpy(buf, &h, sizeof(h));
        Ptr<Packet> p = Create<Packet>(buf, 128);
        if (m_onSend)
        {
            m_onSend();
        }
        m_txSock->SendTo(p, 0, InetSocketAddress(m_bcast, m_port));
    }
    // seq advances whether or not we were allowed to send: receivers divide by the
    // number that SHOULD have arrived, so a suppressed beacon counts as a loss for
    // the sender's own duty cycle, not as a phantom success.
    m_beaconSeq++;
    Time nxt = (m_nSlots > 1) ? TimeToMySlot() : m_beaconInterval;
    m_beaconEv = Simulator::Schedule(nxt, &MeshNodeApp::SendBeacon, this);
}

void
MeshNodeApp::SendData()
{
    if (m_dataKbps <= 0)
    {
        return;
    }
    if (m_txEnabled && InMySlot())
    {
        MeshPktHdr h{m_myIdx, m_dataSeq, 1, m_slot};
        std::vector<uint8_t> buf(m_pktBytes, 0);
        std::memcpy(buf.data(), &h, sizeof(h));
        Ptr<Packet> p = Create<Packet>(buf.data(), m_pktBytes);
        if (m_onSend)
        {
            m_onSend();
        }
        m_txSock->SendTo(p, 0, InetSocketAddress(m_dataDst, m_port));
        m_dataSeq++;
    }
    double interS = (m_pktBytes * 8.0) / (m_dataKbps * 1000.0);
    Time nxt = Seconds(interS);
    if (m_nSlots > 1 && !InMySlot())
    {
        nxt = TimeToMySlot();   // wait for our slot rather than burning the packet
    }
    m_dataEv = Simulator::Schedule(nxt, &MeshNodeApp::SendData, this);
}

void
MeshNodeApp::HandleRead(Ptr<Socket> s)
{
    Ptr<Packet> p;
    Address from;
    while ((p = s->RecvFrom(from)))
    {
        if (p->GetSize() < sizeof(MeshPktHdr))
        {
            continue;
        }
        uint8_t buf[sizeof(MeshPktHdr)];
        p->CopyData(buf, sizeof(MeshPktHdr));
        MeshPktHdr h;
        std::memcpy(&h, buf, sizeof(h));
        m_rxCount[h.srcIdx]++;
        m_rxTotal[h.srcIdx]++;
        double now = Simulator::Now().GetSeconds();
        m_lastHeard[h.srcIdx] = now;
        if (h.kind == 1)
        {
            m_dataRx++;
        }

        if (h.kind == 0)
        {
            // ---- TX-shadow bucketing (TELEMETRY.md §1) ----
            // Beacons are periodic, so a sequence gap tells us exactly WHEN each
            // missing beacon was due. Classify every due beacon by whether it fell
            // inside the window right after one of our own transmissions.
            auto ls = m_lastSeq.find(h.srcIdx);
            if (ls != m_lastSeq.end() && m_inTxShadow)
            {
                uint32_t prevSeq = ls->second;
                double prevT = m_lastSeqTime[h.srcIdx];
                if (h.seq > prevSeq && (h.seq - prevSeq) < 200)
                {
                    uint32_t gap = h.seq - prevSeq;
                    double step = (gap > 0) ? (now - prevT) / gap : 0.0;
                    m_expTotal[h.srcIdx] += gap;
                    m_rcvTotal[h.srcIdx] += 1;
                    {
                        uint32_t e = m_expTotal[h.srcIdx], r = m_rcvTotal[h.srcIdx];
                        if (e > 400) // keep it a rolling estimate
                        {
                            m_expTotal[h.srcIdx] = e / 2;
                            m_rcvTotal[h.srcIdx] = r / 2;
                            e /= 2;
                            r /= 2;
                        }
                        m_rxPdr[h.srcIdx] = e ? double(r) / double(e) : 1.0;
                    }
                    for (uint32_t k = 1; k <= gap; ++k)
                    {
                        double tDue = prevT + step * k;
                        bool missed = (k < gap); // the last one is the packet we just got
                        if (m_inTxShadow(tDue))
                        {
                            m_shadowExp++;
                            if (missed)
                            {
                                m_shadowMiss++;
                            }
                        }
                        else
                        {
                            m_silentExp++;
                            if (missed)
                            {
                                m_silentMiss++;
                            }
                        }
                    }
                }
            }
            m_lastSeq[h.srcIdx] = h.seq;
            m_lastSeqTime[h.srcIdx] = now;

            // ---- pull out the report about US ----
            if (h.nReports > 0 && p->GetSize() >= sizeof(MeshPktHdr) +
                                                      h.nReports * sizeof(LinkReport))
            {
                std::vector<uint8_t> all(p->GetSize());
                p->CopyData(all.data(), all.size());
                for (uint8_t k = 0; k < h.nReports; ++k)
                {
                    LinkReport r;
                    std::memcpy(&r, all.data() + sizeof(MeshPktHdr) + k * sizeof(LinkReport),
                                sizeof(r));
                    if (r.peerIdx == m_myIdx)
                    {
                        m_reverse[h.srcIdx] =
                            std::make_tuple(double(r.rssiDbm), r.pdrQ / 255.0, now);
                    }
                }
            }
        }
    }
}

} // namespace ns3
