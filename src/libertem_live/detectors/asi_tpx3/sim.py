#!/usr/bin/env python
import socket
import time
import os

import memfd
import click


def send_full_file(cache_fd, total_size, conn):
    os.lseek(cache_fd, 0, 0)
    total_sent = 0
    while total_sent < total_size:
        total_sent += os.sendfile(
            conn.fileno(),
            cache_fd,
            total_sent,
            total_size - total_sent,
        )


def handle_connection(cache_fd, total_size, conn, sleep):
    total_sent_this_conn = 0
    try:
        while True:
            print("sending full file...")
            t0 = time.perf_counter()
            send_full_file(cache_fd, total_size, conn)
            t1 = time.perf_counter()
            thp = total_size / 1024 / 1024 / (t1-t0)
            total_sent_this_conn += total_size
            print(f"done in {t1-t0:.3f}s, throughput={thp:.3f}MiB/s")
            time.sleep(sleep)
            t2 = time.perf_counter()
            thp_with_sleep = total_size / 1024 / 1024 / (t2-t0)
            print(f"throughput_with_sleep={thp_with_sleep:.3f}MiB/s")
    finally:
        print(f"connection closed, total_sent = {total_sent_this_conn/1024/1024/1024:.3f}GiB")


def start_server(cache_fd, total_size, sleep):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 8283))
    sock.settimeout(1)
    sock.listen(1)
    while True:
        try:
            conn, client_addr = sock.accept()
            handle_connection(cache_fd, total_size, conn, sleep)
        except socket.timeout:
            continue
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"{e}")
            continue
        except Exception as e:
            print(f"got some other error: {e}")
            raise


@click.command()
@click.argument(
    'paths',
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True
    ),
    nargs=-1
)
@click.option('--sleep', type=float)
def main(paths, sleep=0.5):
    print("populating cache...")
    cache_fd = memfd.memfd_create("tpx_cache", 0)
    total_size = 0
    for path in paths:
        written = 0
        print(f"reading {path}")
        with open(path, "rb") as f:
            in_bytes = f.read()
            while written < len(in_bytes):
                written += os.write(cache_fd, in_bytes[written:])
        total_size += written
    os.lseek(cache_fd, 0, 0)
    print(f"cache populated, total_size={total_size}")

    start_server(cache_fd, total_size, sleep)


if __name__ == "__main__":
    main()
