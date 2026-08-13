from __future__ import annotations

import math

from .nemotron3_layout import GroupMajorSkewedLayout, ProjectionField, packet_service_cycles


def test_nemotron_projection_mapping_is_bijective_with_small_padding() -> None:
    layout = GroupMajorSkewedLayout()
    addresses = layout.logical_addresses()
    assert len(addresses) == 10304
    assert len(set(addresses.values())) == 10304
    assert layout.physical_values == 10368
    assert layout.padding_values == 64


def test_candidate_packets_meet_bank_lower_bound() -> None:
    layout = GroupMajorSkewedLayout()
    for group in range(8):
        for packet in layout.state_input_packets(group) + layout.gate_packets(group):
            actual = packet_service_cycles(packet, banks=16)
            lower_bound = math.ceil(len(packet) / 16)
            assert actual == lower_bound


def test_bc_packet_is_one_read_per_bank_and_not_expanded_per_head() -> None:
    layout = GroupMajorSkewedLayout()
    bc_packet = layout.state_input_packets(0)[-1]
    assert len(bc_packet) == 16
    assert {address.bank for address in bc_packet} == set(range(16))
    assert [address.bank for address in bc_packet[:8]] == list(range(8, 16))
    assert [address.bank for address in bc_packet[8:]] == list(range(0, 8))


def test_head_skew_spreads_32_x_values_over_two_balanced_beats() -> None:
    layout = GroupMajorSkewedLayout()
    packet = layout.state_input_packets(0)[1]
    assert len(packet) == 32
    assert packet_service_cycles(packet, banks=16) == 2
    counts = [sum(address.bank == bank for address in packet) for bank in range(16)]
    assert counts == [2] * 16


def test_cyclic_skew_changes_bank_without_changing_field_row_ownership() -> None:
    layout = GroupMajorSkewedLayout()
    head0 = layout.address(ProjectionField.X, 0, 0, 0)
    head1 = layout.address(ProjectionField.X, 0, 1, 0)
    assert head0.bank == 0
    assert head1.bank == 4
    assert head1.row - head0.row == 4
