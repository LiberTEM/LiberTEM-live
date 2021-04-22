import contextlib
import socket
import struct


def make_mreqn(group: str, iface: str):
    """
    This function creates a byte array containing an instance of the
    `ip_mreqn` struct:

    struct ip_mreqn {
       struct in_addr imr_multiaddr; /* IP multicast group
                                        address */
       struct in_addr imr_address;   /* IP address of local
                                        interface */
       int            imr_ifindex;   /* interface index */
    };

    - `in_addr` is just a fancy way of saying "an IP address as bytes in network order"
    - the interface index can be obtained by calling `socket.if_nametoindex`

    After joining a multicast group using the mreqn, you can check in
    `/proc/net/igmp` that the right index was used for joining. For example,
    after joining 225.1.1.1 on some interface, you should see an entry
    with Group=010101E1 in `/proc/net/igmp` below that interface.
    """
    group_bin = socket.inet_pton(socket.AF_INET, group)
    local_bin = socket.inet_pton(socket.AF_INET, '0.0.0.0')
    idx = socket.if_nametoindex(iface)
    return struct.pack('4s4sI', group_bin, local_bin, idx)


def bind_mcast_socket(port: int, group: str, local_addr: str, iface: str):
    SO_REUSEPORT = getattr(socket, 'SO_REUSEPORT', 15)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, SO_REUSEPORT, 1)
    mreq = make_mreqn(group, iface)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.bind((group, port))  # group, '', local_addr
    return s


@contextlib.contextmanager
def mcast_socket(port: int, group: str, local_addr: str, iface: str):
    s = bind_mcast_socket(port, group, local_addr, iface)
    try:
        yield s
    finally:
        s.close()
