// ===== FILE: project/src/lib/ThroneLib.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

// ─────────────────────────────────────────────────────────────────────────────
//  ThroneLib · v1.1a · external-linked library for the throne royalty mechanic
//
//  Bytecode size optimization. The main BurnBombV1_1a contract pushes against
//  the EIP-170 24KB deployed-code ceiling once the throne logic + on-chain
//  metadata fields are bundled in. Extracting the throne mutation helpers to
//  an external library frees ~1KB from the main contract bytecode at the cost
//  of one DELEGATECALL hop per throne update (a few thousand gas).
//
//  The library operates on storage references passed by the caller — the
//  throne[3] / throneSize[3] / seasonStartedAt fields live in the main
//  contract's storage. Library functions are declared `external` so they
//  deploy as separate bytecode + are linked at compile time.
//
//  Design semantics encoded here (mirror MINNOW-V1.1A-SPEC.md §2.6 + §2.7):
//    - One-seat-per-creator (biggest pot wins that creator's seat)
//    - Strict `>` tie rule (incumbents keep their seat against equal challengers)
//    - Sort descending by potSize, seat[0] = highest
//    - Three-element bubble sort (deterministic, small)
// ─────────────────────────────────────────────────────────────────────────────

library ThroneLib {
    /// @dev Apply this round's (creator, pot) to the throne. Mutates the
    /// caller's throne + throneSize storage in place. Returns nothing — the
    /// caller emits ThroneSnapshot post-call with the deterministic final
    /// state.
    function update(
        address[3] storage throne,
        uint256[3] storage throneSize,
        address creator,
        uint256 newPot
    ) external {
        // Path A — creator already holds a seat
        for (uint256 i = 0; i < 3; ) {
            if (throne[i] == creator) {
                if (newPot > throneSize[i]) {
                    throneSize[i] = newPot;
                    _resort(throne, throneSize);
                }
                return;
            }
            unchecked { ++i; }
        }
        // Path B — creator not on throne; try displacement of smallest seat
        if (newPot > throneSize[2]) {
            throne[2] = creator;
            throneSize[2] = newPot;
            _resort(throne, throneSize);
        }
    }

    /// @dev Wipe the throne to empty (used by season reset). The caller emits
    /// SeasonReset around this call.
    function wipe(
        address[3] storage throne,
        uint256[3] storage throneSize
    ) external {
        throne[0] = address(0); throne[1] = address(0); throne[2] = address(0);
        throneSize[0] = 0; throneSize[1] = 0; throneSize[2] = 0;
    }

    /// @dev Distribute the thronePool across the current throne (post-update)
    /// in the operator-locked 70/20/10 split. Empty seats return their share
    /// as the third return arg so caller can route to winner. Residual-to-
    /// seat-3 prevents dust loss (matches v1.1's winnerShare-as-residual).
    ///
    /// Returns:
    ///   credited: total credited across non-empty seats (not strictly needed
    ///             but useful for caller assertions/tests)
    ///   emptySeatOverflow: total share that had no eligible seat owner
    ///   updatedPayout: side-effect via the storage mapping reference
    function payout(
        address[3] storage throne,
        uint256 thronePool,
        uint256 roundId,
        mapping(uint256 => mapping(address => uint256)) storage claimablePayout
    ) external returns (uint256 credited, uint256 emptySeatOverflow) {
        uint256 seat1Pay = thronePool * 70 / 100;
        uint256 seat2Pay = thronePool * 20 / 100;
        uint256 seat3Pay = thronePool - seat1Pay - seat2Pay; // residual

        if (throne[0] != address(0)) {
            claimablePayout[roundId][throne[0]] += seat1Pay;
            credited += seat1Pay;
        } else {
            emptySeatOverflow += seat1Pay;
        }
        if (throne[1] != address(0)) {
            claimablePayout[roundId][throne[1]] += seat2Pay;
            credited += seat2Pay;
        } else {
            emptySeatOverflow += seat2Pay;
        }
        if (throne[2] != address(0)) {
            claimablePayout[roundId][throne[2]] += seat3Pay;
            credited += seat3Pay;
        } else {
            emptySeatOverflow += seat3Pay;
        }
    }

    /// @dev Internal helper · three-element bubble sort descending. Strict `>`
    /// only (incumbent-favoring tie rule).
    function _resort(
        address[3] storage throne,
        uint256[3] storage throneSize
    ) internal {
        if (throneSize[1] > throneSize[0]) {
            (throne[0], throne[1]) = (throne[1], throne[0]);
            (throneSize[0], throneSize[1]) = (throneSize[1], throneSize[0]);
        }
        if (throneSize[2] > throneSize[1]) {
            (throne[1], throne[2]) = (throne[2], throne[1]);
            (throneSize[1], throneSize[2]) = (throneSize[2], throneSize[1]);
        }
        if (throneSize[1] > throneSize[0]) {
            (throne[0], throne[1]) = (throne[1], throne[0]);
            (throneSize[0], throneSize[1]) = (throneSize[1], throneSize[0]);
        }
    }
}
