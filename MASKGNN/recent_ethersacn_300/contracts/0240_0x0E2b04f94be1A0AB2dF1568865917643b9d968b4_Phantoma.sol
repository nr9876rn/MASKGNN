// ===== FILE: src/Phantoma.sol =====
// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

//
//   ✿ ✧ ★ ♡ ❀ ✦ ★ ♥ ✿ ★ ♡ ✧ ❀ ♥ ★ ✦ ✿ ♡ ★ ❀ ♥ ✧ ★ ♡ ✦
//
//          ____  _                 _
//         |  _ \| |__   __ _ _ __ | |_ ___  _ __ ___   __ _
//         | |_) | '_ \ / _` | '_ \| __/ _ \| '_ ` _ \ / _` |
//         |  __/| | | | (_| | | | | || (_) | | | | | | (_| |
//         |_|   |_| |_|\__,_|_| |_|\__\___/|_| |_| |_|\__,_|
//
//        ⌒∿⌒  ✧  ♥  ✦  ★  ✧  ♥  ✦  ★  ✧  ♥  ✦  ★  ⌒∿⌒
//
//   9,998 rigged 3D characters bijective with Milady Maker.
//   Identity sacred.
//   Built to be built on.
//
//              ❀  ✧  ♥  LordThisDrip · 2026  ♥  ✧  ❀
//
//   ✿ ✧ ★ ♡ ❀ ✦ ★ ♥ ✿ ★ ♡ ✧ ❀ ♥ ★ ✦ ✿ ♡ ★ ❀ ♥ ✧ ★ ♡ ✦
//

import {ERC721} from "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import {ERC2981} from "@openzeppelin/contracts/token/common/ERC2981.sol";
import {IERC2981} from "@openzeppelin/contracts/interfaces/IERC2981.sol";
import {ReentrancyGuardTransient} from "@openzeppelin/contracts/utils/ReentrancyGuardTransient.sol";
import {Strings} from "@openzeppelin/contracts/utils/Strings.sol";
import {IERC6551Registry} from "./interfaces/IERC6551.sol";
import {IERC6551Equipment} from "./interfaces/IERC6551Equipment.sol";

/// @dev Minimal subset of PhantomaCosmetics used by Phantoma. Adding methods
///      here does not require changing the deployed PhantomaCosmetics — the
///      real contract already exposes these getters. Only the methods Phantoma
///      actively calls live on this interface; full surface lives on the
///      concrete contract.
interface IPhantomaCosmetics {
    function mintFounderPack(address minter, uint256 mintSequence, uint8 phase, bool isBijective) external;
    function tierCount() external view returns (uint256);
    function tierItemIdsAt(uint256 t) external view returns (uint256[] memory);
    function balanceOf(address account, uint256 id) external view returns (uint256);
    function protocolTransfer(address from, address to, uint256 id, uint256 amount) external;
}

/// @dev Subset of `EquippableAccount` used by Phantoma's cosmetic-transfer
///      and (forthcoming) marketplace paths. Lives outside the canonical
///      `IERC6551Equipment` interface to preserve the ERC-8216 ID
///      `0xc1ef0b9e` pinned by `test/IERC6551Equipment.interfaceId.t.sol`.
interface IEquippableAccountExtended {
    function maxSlotReservation(address tokenContract, uint256 tokenId) external view returns (uint256);
}

interface IERC721Minimal {
    function ownerOf(uint256 tokenId) external view returns (address);
}

interface IERC20Minimal {
    function balanceOf(address account) external view returns (uint256);
}

/// @title Phantoma
/// @notice ERC-721 collection with bijective source-collection claim phase, commit-reveal
///         random-draw phases, lazy ERC-6551 TBA resolution, and bounded-mutability token URIs.
/// @author LordThisDrip
contract Phantoma is ERC721, ERC2981, ReentrancyGuardTransient {
    // ---------------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------------
    /// @notice Total Phantoma supply. Mirrors the Milady Maker token ID space
    ///         [0, 9999]. If specific Milady IDs are unminted or burned, the
    ///         corresponding Phantoma IDs are unclaimable via Phase 1 and flow
    ///         into the Phase 2/3 random-draw pool instead.
    uint256 public constant MILADY_SUPPLY      = 10000;
    uint256 public constant PHASE_1_DURATION   = 48 hours;
    uint256 public constant PHASE_2_DURATION   = 36 hours;
    uint256 public constant PHASE_1_PRICE      = 0.01 ether;
    uint256 public constant PHASE_2_PRICE      = 0.015 ether;
    uint256 public constant PHASE_3_PRICE      = 0.02 ether;
    /// @notice Cap on `claim`'s `sourceTokenIds.length`. Set so that a
    ///         worst-case Medallion-cohort batch (full Path-B mint at
    ///         ~1.022M gas/mint) fits comfortably under mainnet's ~30M
    ///         practical block-gas target: 25 x 1.022M = 25.55M, leaving
    ///         margin for gas-price spikes and block fullness. The previous
    ///         50-cap could not be honored for Medallion-eligible batches
    ///         (51M required) — friend-audit-quality nit closed by Pass 6.
    uint256 public constant MAX_BATCH_CLAIM    = 25;
    /// @notice Cap on `commitMany`'s and `revealMany`'s array lengths.
    ///         Set to 10 to match `PHASE_X_PER_WALLET` — per-wallet cap
    ///         is the natural ceiling since attempting more would revert
    ///         `WalletLimitExceeded` at the cap check anyway. Gas envelope
    ///         at the cap is comfortable: revealMany(10) worst-case Phase 3
    ///         bijective ≈ 7M gas, well under the 30M block-gas target.
    ///         Pass 18 (2026-05-21).
    uint256 public constant MAX_BATCH_COMMIT   = 10;
    uint256 public constant MAX_BATCH_REVEAL   = 10;
    uint256 public constant PHASE_2_PER_WALLET = 10;
    uint256 public constant PHASE_3_PER_WALLET = 10;
    uint256 public constant GRACE_PERIOD       = 120 hours;
    uint8   public constant MAX_SPINS          = 3;
    uint256 public constant REVEAL_WINDOW_MIN  = 2;
    uint256 public constant REVEAL_WINDOW_MAX  = 250;
    uint96  public constant ROYALTY_BPS        = 500;
    string  public constant PLACEHOLDER_URI    = "ipfs://QmZvHXMDCDk2WLK637HbQHNWu8iPXGWnessRgFuAfurgbX";
    uint256 public constant CULT_THRESHOLD     = 2_500_000e18;

    /// @notice Marketplace listing-duration bounds.
    /// @dev MAX prevents indefinite state growth from forgotten listings.
    ///      MIN prevents pump-and-cancel grief patterns (instant relist
    ///      front-running). Both fit uint64 trivially.
    uint64 public constant MAX_LISTING_DURATION = 30 days;
    uint64 public constant MIN_LISTING_DURATION = 1 hours;

    /// @notice Slot key used by Phantoma to anchor S-2 on every TBA via a
    ///         locked binding-anchor cosmetic (Hub Key on mainnet,
    ///         Director-locked 2026-05-01). Dedicated — the user does not
    ///         interact with this slot. Pass 4 OB-2 (Path B).
    bytes32 public constant BINDING_SLOT   = keccak256("phantoma.slot.binding");
    /// @notice Slot key used by Phantoma to lock the Medallion when the
    ///         recipient qualifies for the rarest tier. Available for
    ///         user `equipFromBalance` for non-Medallion-tier mints.
    ///         Pass 4 OB-2 (Path B).
    bytes32 public constant ACCESSORY_SLOT = keccak256("phantoma.slot.accessory");

    // ---------------------------------------------------------------------
    // Immutables
    // ---------------------------------------------------------------------
    address public immutable miladyContract;
    address public immutable cultToken;
    address public immutable tbaRegistry;
    address public immutable equippableAccountImpl;
    address public immutable cosmeticsContract;
    address public immutable pipelineOperator;
    address public immutable treasury;
    address public immutable launcher;
    /// @notice Cached `tierCount() - 1` from PhantomaCosmetics, computed at
    ///         construction time. The catch-all tier (and therefore the
    ///         binding-slot anchor item) lives at this index by the
    ///         constructor invariant in PhantomaCosmetics
    ///         (`CatchAllNotLastTier` revert if violated). Caching as an
    ///         immutable saves a `tierCount()` external call per mint —
    ///         ~3k gas × 10,000 mints if read live each time.
    uint256 public immutable cosmeticsCatchAllTier;

    /// @notice Cached `tierItemIdsAt(cosmeticsCatchAllTier)[0]` — the
    ///         **canonical binding-slot anchor item ID**. Mainnet config
    ///         (Director-locked 2026-05-01): Hub Key (1006), the universal
    ///         identity-marker cosmetic minted to every TBA and locked into
    ///         `phantoma.slot.binding` at mint. Pre-Hub-Key this was the
    ///         FoodStamp ID; after the Hub Key swap, FoodStamp is no longer
    ///         the binding anchor (FoodStamp ×4 mints loose, none locked).
    ///
    ///         Pinned at deploy time by `PhantomaCosmetics`'s immutable
    ///         tier-config arrays + the `CatchAllNotLastTier` constructor
    ///         invariant. Saves an external `tierItemIdsAt(...)` STATICCALL
    ///         (and the array decode) on every mint.
    ///
    ///         Consumer pattern (frontends, indexers, marketplaces): for
    ///         trust-validation of a Phantoma's binding-slot occupant, read
    ///         `hubKeyId()` at runtime and assert the binding-slot entry's
    ///         tokenId matches. The runtime-read pattern is future-proof
    ///         against further anchor swaps — the immutable's CONCRETE
    ///         purpose is "the protocol-locked binding anchor item ID,"
    ///         regardless of which cosmetic that currently is. Hardcoding
    ///         the numeric ID into consumer logic is a maintenance hazard.
    uint256 public immutable hubKeyId;

    /// @notice Cached `tierItemIdsAt(0)[0]` — the Medallion item ID. Tier 0
    ///         is the rarest tier per the ascending-thresholds invariant in
    ///         `PhantomaCosmetics`. The Medallion equip+lock at mint is
    ///         conditional on `balanceOf(tba, medallionId) > 0`; the probe
    ///         uses this immutable instead of re-fetching the item ID.
    uint256 public immutable medallionId;

    // ---------------------------------------------------------------------
    // Storage: sparse set (Fix A — +1 offset on positionOf)
    // ---------------------------------------------------------------------
    mapping(uint256 => uint256) private unclaimedIds;
    mapping(uint256 => uint256) private positionOf;
    uint256 public unclaimedCount;

    // ---------------------------------------------------------------------
    // Storage: mint state
    // ---------------------------------------------------------------------
    uint256 public mintCount;
    uint256 public phase1Start;

    // ---------------------------------------------------------------------
    // Storage: commit-reveal
    // ---------------------------------------------------------------------
    struct Commit {
        address committer;
        uint96  price;
        uint256 commitBlock;
        uint8   phase;
    }
    mapping(bytes32 => Commit) public commits;
    uint256 public totalActiveCommitEscrow;

    /// @notice Per-wallet, per-phase counters packed into a single 32-byte
    ///         storage slot. The four uint64 fields exactly fill 32 bytes:
    ///         {phase2Active, phase2Minted, phase3Active, phase3Minted}.
    ///         Each phase's reveal-time update writes both `active` and
    ///         `minted` for that phase in a single SSTORE rather than two —
    ///         saves ~5k gas per reveal × ~8K Phase 2/3 reveals (Pass 5 gas
    ///         optimization, Item A).
    ///
    ///         uint64 max = 1.8 × 10^19, vastly larger than the per-wallet
    ///         cap of 10. Chosen for packing density (4 × 8 bytes = 32) and
    ///         word alignment, not for range. Runtime arithmetic promotes
    ///         to uint256 when comparing against `PHASE_X_PER_WALLET`.
    ///
    ///         The cap semantics are unchanged from HIGH-1A:
    ///           - commit increments `phaseXActive` after the cap check
    ///             `phaseXActive + phaseXMinted < PHASE_X_PER_WALLET`.
    ///           - successful reveal decrements `phaseXActive` and
    ///             increments `phaseXMinted` (one SSTORE for the pair).
    ///           - successful expire decrements `phaseXActive`.
    ///         The reveal-time `phaseXMinted >= PHASE_X_PER_WALLET` check
    ///         is retained as defense in depth.
    ///
    ///         Backward-compatible ABI is preserved via four explicit
    ///         wrapper getters (`phase2Active(addr)` etc.) defined below.
    struct WalletCounters {
        uint64 phase2Active;
        uint64 phase2Minted;
        uint64 phase3Active;
        uint64 phase3Minted;
    }
    mapping(address => WalletCounters) private _walletCounters;

    // -- Backward-compatible getters preserving the pre-packing ABI ---------

    /// @notice Pre-packing ABI shim. Returns this wallet's currently-active
    ///         (committed but not yet revealed or refunded) Phase 2 commit
    ///         count.
    function phase2Active(address wallet) external view returns (uint256) {
        return uint256(_walletCounters[wallet].phase2Active);
    }

    /// @notice Pre-packing ABI shim. Returns this wallet's successfully
    ///         revealed Phase 2 mint count.
    function phase2Minted(address wallet) external view returns (uint256) {
        return uint256(_walletCounters[wallet].phase2Minted);
    }

    /// @notice Pre-packing ABI shim. Returns this wallet's currently-active
    ///         Phase 3 commit count.
    function phase3Active(address wallet) external view returns (uint256) {
        return uint256(_walletCounters[wallet].phase3Active);
    }

    /// @notice Pre-packing ABI shim. Returns this wallet's successfully
    ///         revealed Phase 3 mint count.
    function phase3Minted(address wallet) external view returns (uint256) {
        return uint256(_walletCounters[wallet].phase3Minted);
    }

    // ---------------------------------------------------------------------
    // Storage: URI state
    // ---------------------------------------------------------------------
    mapping(uint256 => string)  private _tokenURIs;
    mapping(uint256 => uint256) public  resolvedAt;
    mapping(uint256 => uint8)   public  spinCount;
    mapping(uint256 => bool)    public  tokenURISet;
    mapping(uint256 => bool)    public  tokenFinalized;
    string private _phantomaBaseURI;
    bool public baseURILocked;

    // ---------------------------------------------------------------------
    // Storage: marketplace (Layer 2 — listings)
    // ---------------------------------------------------------------------
    /// @notice A Phantoma marketplace listing. Stored under sequential
    ///         listingIds in the `listings` mapping.
    /// @dev    Layout: 4 storage slots per listing (slot 1: sellerTokenId,
    ///         slot 2: cosmeticId, slot 3: amount, slot 4: price+expiration
    ///         packed — 12+8 = 20 bytes used, 12 bytes spare).
    ///
    ///         `sellerTokenId` is the Phantoma whose TBA holds the cosmetic
    ///         being sold. The listing's authority + payment routing both
    ///         resolve to `_ownerOf(sellerTokenId)` AT QUERY TIME — supports
    ///         the documented listing-inheritance semantic where selling
    ///         the Phantoma transfers the right-to-cancel and the right-to-
    ///         receive-proceeds for any active listing on its TBA.
    ///
    ///         `price` is uint96 — caps at ~7.9e10 ETH, plenty for any
    ///         realistic listing. `expiration` is uint64 — caps at ~year
    ///         20450 in unix-time. Both fit in one slot.
    struct Listing {
        uint256 sellerTokenId;
        uint256 cosmeticId;
        uint256 amount;
        uint96  price;
        uint64  expiration;
    }
    mapping(uint256 => Listing) public listings;
    /// @notice Monotonic counter assigning unique IDs to listings.
    ///         Starts at 0; first listing gets ID 1.
    uint256 public listingCounter;

    // ---------------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------------
    event Finalized(uint256 phase1Start, uint256 unclaimedCount);
    /// @notice Emitted once per phase at finalize time — indexers can
    ///         reconstruct phase boundaries from the event stream alone
    ///         without knowledge of the contract's duration constants.
    ///         Covers spec §17 Utopia Principle point 6 for phase transitions.
    event PhaseStarted(uint8 indexed phase, uint256 startTimestamp, uint256 endTimestamp);
    event PhantomaMinted(
        address indexed to,
        uint256 indexed tokenId,
        uint256 indexed mintSequence,
        uint8 phase
    );
    event CommitSubmitted(
        bytes32 indexed commitHash,
        address indexed committer,
        uint8 phase,
        uint256 commitBlock,
        uint256 price
    );
    /// @notice Emitted on successful reveal. `seed` and `totalMinted` are
    ///         included so indexers can verify mint selection from events
    ///         alone — `seed % unclaimedCount_at_reveal == selectedIndex`
    ///         reproduces the contract's draw. Per spec §17 Utopia Principle.
    event CommitRevealed(
        bytes32 indexed commitHash,
        address indexed committer,
        uint256 indexed tokenId,
        uint256 seed,
        uint256 totalMinted
    );
    /// @notice Emitted when an expired commit is closed via `expireCommit`.
    ///         Per HIGH-1B, the commit's escrow is forfeited to the protocol
    ///         (it remains in the contract balance and becomes sweepable to
    ///         treasury via the existing `sweep()` reserve mechanism). The
    ///         `price` field reports the forfeited amount for indexer
    ///         accounting; no transfer is made to the committer.
    event CommitExpired(
        bytes32 indexed commitHash,
        address indexed committer,
        uint256 price
    );
    event TokenURIResolved(uint256 indexed tokenId, string uri, uint256 timestamp);
    event TokenURIUpdatedDuringGrace(
        uint256 indexed tokenId,
        string oldURI,
        string newURI,
        uint8 spinNumber,
        uint256 timestamp
    );
    event TokenFinalized(uint256 indexed tokenId, uint256 timestamp);
    event Swept(address indexed treasury, uint256 amount);
    event BaseURISet(string baseURI);
    event BaseURILocked(string baseURI);

    /// @notice ERC-4906 single-token metadata update event. Marketplaces
    ///         (OpenSea / Blur / Magic Eden) listen for this event and
    ///         auto-refresh cached metadata for the affected tokenId.
    ///
    ///         Emitted at every state transition that alters the
    ///         composite metadata of a Phantoma:
    ///           - `setTokenURI(tokenId, ...)` (per-token URI override
    ///             by Pipeline operator within grace window)
    ///           - `_executeTransfer(...)` (cosmetic moves between TBAs
    ///             via marketplace purchase or free gift — fires for
    ///             BOTH source and destination Phantoma)
    ///           - `emitMetadataUpdate(tokenId)` (TBA-driven slot state
    ///             changes: equip / unequip / lockSlot via the
    ///             EquippableAccount callback path)
    ///
    ///         Event signature matches EIP-4906 exactly. Underscore-
    ///         prefixed parameter name is the spec convention.
    event MetadataUpdate(uint256 _tokenId);

    /// @notice ERC-4906 batch metadata update event. Emitted by
    ///         `setBaseURI` for the full Phantoma token id range
    ///         [0, MILADY_SUPPLY-1] = [0, 9999]. Marketplaces refresh
    ///         all minted tokens in the range.
    event BatchMetadataUpdate(uint256 _fromTokenId, uint256 _toTokenId);
    /// @notice Emitted on every successful `transferCosmeticBetweenTBAs`
    ///         (free gift path) and on the cosmetic-transfer leg of a
    ///         marketplace purchase. Surfaces the tokenId-pair semantic
    ///         for indexer convenience — the underlying ERC-1155
    ///         `TransferSingle` and `ProtocolTransfer` events fire on
    ///         PhantomaCosmetics with the TBA addresses.
    event CosmeticTransferred(
        uint256 indexed fromTokenId,
        uint256 indexed toTokenId,
        uint256 indexed cosmeticId,
        uint256 amount,
        address sender
    );

    /// @notice Emitted when a Phantoma owner creates a marketplace listing.
    ///         Frontends index this event to build the active-listing UI;
    ///         no on-chain enumeration is provided.
    event CosmeticListed(
        uint256 indexed listingId,
        uint256 indexed sellerTokenId,
        uint256 indexed cosmeticId,
        uint256 amount,
        uint96  price,
        uint64  expiration,
        address lister
    );
    /// @notice Emitted when a listing is cancelled by the current owner of
    ///         `listing.sellerTokenId` (supports listing-inheritance: if the
    ///         Phantoma was sold while the listing was active, the new owner
    ///         can cancel).
    event ListingCancelled(uint256 indexed listingId, address canceller);
    /// @notice Emitted on a successful marketplace purchase. Surfaces the
    ///         tokenId-pair semantic plus payment + royalty breakdown for
    ///         indexers building marketplace activity feeds.
    event CosmeticPurchased(
        uint256 indexed listingId,
        uint256 indexed sellerTokenId,
        uint256 indexed buyerTokenId,
        uint256 cosmeticId,
        uint256 amount,
        uint256 price,
        uint256 royaltyAmount
    );

    // ---------------------------------------------------------------------
    // Errors
    // ---------------------------------------------------------------------
    error NotLauncher();
    error NotPipelineOperator();
    error NotTreasury();
    error AlreadyFinalized();
    error NotYetFinalized();
    error ZeroAddress();
    error BaseURIIsLocked();

    error WrongPhase(uint8 expected, uint8 actual);
    /// @notice Raised by `claim` when the contract is finalized but the
    ///         current `block.timestamp` is still strictly less than
    ///         `phase1Start`. Distinct from `WrongPhase(1, 0)` so callers
    ///         can disambiguate "claim outside phase 1" (irrelevant under
    ///         the multi-phase claim semantic) from "claim before any
    ///         phase has begun" (the only state where `_currentPhase()`
    ///         returns 0 post-finalize).
    error PhaseNotActive();
    error IncorrectPayment(uint256 sent, uint256 required);
    error InvalidPhaseStart();

    error EmptyBatch();
    error BatchTooLarge(uint256 size);
    error IdOutOfRange(uint256 id);
    error AlreadyClaimed(uint256 id);
    error NotMiladyOwner(uint256 id, address caller);
    error DuplicateInBatch(uint256 id);

    error CommitExists(bytes32 commitHash);
    error CommitNotFound(bytes32 commitHash);
    error RevealTooEarly(uint256 currentBlock, uint256 earliest);
    error RevealTooLate(uint256 currentBlock, uint256 latest);
    error CommitNotExpired(uint256 currentBlock, uint256 expiresAt);
    error BlockhashUnavailable(uint256 targetBlock);
    error InvalidReveal();
    error CultGateNotMet(address committer, uint256 balance);
    error WalletLimitExceeded(address wallet, uint8 phase);
    /// @dev Pass 18: commitMany hash-duplicate-within-batch.
    error DuplicateCommitInBatch(bytes32 commitHash);
    /// @dev Pass 18: revealMany requires `secrets.length == nonces.length`.
    error ArrayLengthMismatch(uint256 secretsLen, uint256 noncesLen);
    error PoolEmpty();

    error TokenURINotSet(uint256 tokenId);
    error TokenAlreadyFinalized(uint256 tokenId);
    error GracePeriodExpired(uint256 tokenId);
    error SpinLimitExceeded(uint256 tokenId);

    error SweepUnderflow(uint256 balance, uint256 reserve);
    error SweepFailed(address treasury, uint256 amount);

    // -------- Cosmetic transfer / marketplace errors --------
    /// @notice Raised when caller attempts to transfer/list/purchase with
    ///         the same source and destination Phantoma. Self-moves are
    ///         no-ops at best, footguns at worst.
    error SelfTransfer();
    /// @notice Raised when caller attempts a cosmetic operation with
    ///         `amount == 0`.
    error InvalidAmount();
    /// @notice Raised when `msg.sender` is not the current owner of
    ///         `tokenId` for an ownership-gated cosmetic operation
    ///         (transferCosmeticBetweenTBAs, listCosmetic, cancelListing,
    ///         purchaseCosmetic's buyer side).
    error NotPhantomaOwner(uint256 tokenId, address caller);
    /// @notice Raised when the source TBA does not hold enough of the
    ///         requested cosmetic for a transfer.
    error InsufficientCosmeticBalance(
        address tba,
        uint256 cosmeticId,
        uint256 requested,
        uint256 available
    );
    /// @notice Raised when the requested transfer would leave the source
    ///         TBA's balance below the maximum single-slot reservation
    ///         for the cosmetic — i.e., would break an equipped slot's
    ///         S-1 invariant. Locked items (Hub Key in binding, Medallion
    ///         in accessory for Medallion-tier mints) always trigger this
    ///         on any transfer attempt; equipped non-locked items trigger
    ///         it unless the caller unequips first or transfers only the
    ///         surplus above the slot reservation.
    error CosmeticReservedInSlot(
        address tba,
        uint256 cosmeticId,
        uint256 attemptedTransfer,
        uint256 balance,
        uint256 maxSlotReservation
    );

    // -------- Marketplace errors (Layer 2) --------
    /// @notice Raised when caller passes `price == 0` to `listCosmetic`.
    error InvalidPrice();
    /// @notice Raised when listing duration is below `MIN_LISTING_DURATION`.
    error ListingDurationTooShort(uint64 duration);
    /// @notice Raised when listing duration is above `MAX_LISTING_DURATION`.
    error ListingDurationTooLong(uint64 duration);
    /// @notice Raised by cancelListing / purchaseCosmetic when the listingId
    ///         points at an empty slot (never created, already purchased,
    ///         or already cancelled).
    error ListingNotFound(uint256 listingId);

    // -------- Marketplace errors (Layer 3 — purchase) --------
    /// @notice Raised by purchaseCosmetic when the listing has expired
    ///         (`block.timestamp > listing.expiration`).
    error ListingExpired(uint256 listingId, uint64 expiration);
    /// @notice Raised by purchaseCosmetic when `msg.value != listing.price`
    ///         (over- or under-payment).
    error IncorrectListingPrice(uint256 listingId, uint256 sent, uint256 required);
    /// @notice Raised by purchaseCosmetic when buyerTokenId equals the
    ///         listing's sellerTokenId. Forbidden — pay yourself for your
    ///         own listing is a footgun, not a feature.
    error SelfPurchase();
    /// @notice Raised by purchaseCosmetic when a push-payment to the
    ///         royalty receiver or seller payee fails (recipient contract
    ///         reverts in receive/fallback). Reverts the entire purchase
    ///         atomically; buyer's ETH is EVM-revert-protected.
    error PaymentFailed(address recipient);

    // -------- ERC-4906 callback errors (Layer 5 — metadata refresh) --------
    /// @notice Raised by `emitMetadataUpdate` when `msg.sender` is not
    ///         the canonical TBA address derived for the given `tokenId`.
    ///         Auth gate: msg.sender == _tbaAddressOf(tokenId). Cannot
    ///         be spoofed — the canonical 6551 registry's CREATE2
    ///         derivation is deterministic and uncontrollable by
    ///         external callers.
    error NotTokenBoundAccount(uint256 tokenId, address caller);

    // ---------------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------------
    modifier onlyLauncher()         { if (msg.sender != launcher)         revert NotLauncher();         _; }
    modifier onlyPipelineOperator() { if (msg.sender != pipelineOperator) revert NotPipelineOperator(); _; }
    modifier onlyTreasury()         { if (msg.sender != treasury)         revert NotTreasury();         _; }
    modifier whenFinalized()        { if (phase1Start == 0)               revert NotYetFinalized();     _; }

    // ---------------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------------
    constructor(
        address miladyContract_,
        address cultToken_,
        address tbaRegistry_,
        address equippableAccountImpl_,
        address cosmeticsContract_,
        address pipelineOperator_,
        address treasury_,
        address launcher_
    ) ERC721("Phantoma", "PHNT") {
        if (miladyContract_        == address(0)) revert ZeroAddress();
        if (cultToken_             == address(0)) revert ZeroAddress();
        if (tbaRegistry_           == address(0)) revert ZeroAddress();
        if (equippableAccountImpl_ == address(0)) revert ZeroAddress();
        if (cosmeticsContract_     == address(0)) revert ZeroAddress();
        if (pipelineOperator_      == address(0)) revert ZeroAddress();
        if (treasury_              == address(0)) revert ZeroAddress();
        if (launcher_              == address(0)) revert ZeroAddress();

        miladyContract        = miladyContract_;
        cultToken             = cultToken_;
        tbaRegistry           = tbaRegistry_;
        equippableAccountImpl = equippableAccountImpl_;
        cosmeticsContract     = cosmeticsContract_;
        pipelineOperator      = pipelineOperator_;
        treasury              = treasury_;
        launcher              = launcher_;

        // Cache the catch-all tier index AND the per-tier item IDs Phantoma
        // reads on every mint. PhantomaCosmetics's tier config is set in
        // its constructor from immutable arrays, so these values are pinned
        // for the lifetime of both contracts. Caching avoids three external
        // STATICCALLs per `_mintInternal` invocation. Together this saves
        // ~10-12k gas per mint — across 10,000 mints, ~100M gas.
        //
        // Reading at construction time also fail-fast verifies that the
        // cosmetics contract is live and exposes the expected tier-config
        // surface. A misconfigured `cosmeticsContract_` would revert here
        // (function not found, or empty `tierItemIdsAt(...)` return).
        IPhantomaCosmetics ck = IPhantomaCosmetics(cosmeticsContract_);
        cosmeticsCatchAllTier = ck.tierCount() - 1;
        hubKeyId = ck.tierItemIdsAt(cosmeticsCatchAllTier)[0];
        medallionId = ck.tierItemIdsAt(0)[0];

        _setDefaultRoyalty(treasury_, ROYALTY_BPS);
    }

    // ---------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------
    /// @notice One-shot finalization entrypoint. Callable exactly once by the launcher.
    /// @dev `phase1Start != 0` doubles as the inert flag; no separate boolean.
    ///      Guards: phase1Start_ must be in the interval [block.timestamp,
    ///      block.timestamp + 30 days]. Past-dated values are rejected to
    ///      prevent a launcher mistake (or compromised key) from skipping
    ///      Phase 1 and permanently denying Milady holders their bijective
    ///      claim rights. Far-future values are rejected per D-3 to prevent
    ///      accidental multi-year inertia.
    function finalize(uint256 phase1Start_) external onlyLauncher {
        if (
            phase1Start_ < block.timestamp ||
            phase1Start_ > block.timestamp + 30 days
        ) revert InvalidPhaseStart();
        if (phase1Start != 0) revert AlreadyFinalized();
        phase1Start = phase1Start_;
        unclaimedCount = MILADY_SUPPLY;
        emit Finalized(phase1Start_, MILADY_SUPPLY);

        // Spec §17 Utopia Principle: phase boundaries in the event stream.
        uint256 p1End = phase1Start_ + PHASE_1_DURATION;
        uint256 p2End = p1End + PHASE_2_DURATION;
        emit PhaseStarted(1, phase1Start_, p1End);
        emit PhaseStarted(2, p1End, p2End);
        emit PhaseStarted(3, p2End, type(uint256).max);
    }

    // ---------------------------------------------------------------------
    // Phase helpers
    // ---------------------------------------------------------------------
    function _currentPhase() internal view returns (uint8) {
        if (phase1Start == 0) return 0;
        uint256 nowTs = block.timestamp;
        if (nowTs < phase1Start) return 0;
        if (nowTs < phase1Start + PHASE_1_DURATION) return 1;
        if (nowTs < phase1Start + PHASE_1_DURATION + PHASE_2_DURATION) return 2;
        return 3;
    }

    function currentPhase() external view returns (uint8) {
        return _currentPhase();
    }

    // ---------------------------------------------------------------------
    // Phase 1: claim
    // ---------------------------------------------------------------------
    /// @notice Bijective claim — the Milady #N owner can claim Phantoma #N at
    ///         `PHASE_1_PRICE` in any active phase (1, 2, or 3). Bonus-cosmetic
    ///         eligibility for tiers flagged in `tierBijectiveBonus` is granted
    ///         to bijective claimers regardless of `tierMaxPhase`. Phase 2's
    ///         CULT gate (which lives only in `commit`) does not apply — Milady
    ///         ownership at the moment of claim is sufficient authorization.
    /// @dev    Reverts the entire batch on any invalid id (already-claimed,
    ///         not-owned, duplicate, out-of-range). Uses Fix A sparse-set
    ///         direct removal. The current phase is read once and passed to
    ///         `_mintInternal` so the cosmetic-tier eligibility check in
    ///         `mintFounderPack` correctly gates Tier 0 / Tier 1 / Tier 2 /
    ///         Tier 3 by the actual phase, not Phase 1. `isBijective=true`
    ///         is hardcoded — every `claim` is a bijective Milady claim by
    ///         construction (the ownership check at line ~430 enforces it).
    function claim(uint256[] calldata sourceTokenIds) external payable whenFinalized {
        uint8 currentPh = _currentPhase();
        // _currentPhase() returns 0 only when phase1Start == 0 (impossible
        // under whenFinalized) or block.timestamp < phase1Start. Reject the
        // post-finalize, pre-Phase-1 window. All three active phases (1, 2, 3)
        // are valid claim phases under the Director-locked multi-phase
        // bijective-claim design.
        if (currentPh == 0) revert PhaseNotActive();

        uint256 n = sourceTokenIds.length;
        if (n == 0) revert EmptyBatch();
        if (n > MAX_BATCH_CLAIM) revert BatchTooLarge(n);

        // Phase-standard pricing per Director-locked Option B design
        // (pricing amended 2026-05-14 Pass 15: 70% reduction across phases for
        // launch accessibility; further amended 2026-05-21 Pass 17: Phase 2
        // and Phase 3 reduced for accessibility tuning, Phase 1 unchanged):
        //   Phase 1: PHASE_1_PRICE (0.01 ETH)  — Milady-holder claim price
        //   Phase 2: PHASE_2_PRICE (0.015 ETH) — same price as commit-reveal
        //            Phase 2 mints; no price arbitrage
        //   Phase 3: PHASE_3_PRICE (0.02 ETH)  — same price as commit-reveal
        //            Phase 3 mints
        // Bijective claim is differentiated only by item-set (bonus
        // cosmetics via tierBijectiveBonus) and authorization (Milady
        // ownership replaces CULT gate / phase gate), not by price.
        uint256 phasePrice;
        if (currentPh == 1) {
            phasePrice = PHASE_1_PRICE;
        } else if (currentPh == 2) {
            phasePrice = PHASE_2_PRICE;
        } else {
            phasePrice = PHASE_3_PRICE;
        }
        uint256 required = phasePrice * n;
        if (msg.value != required) revert IncorrectPayment(msg.value, required);

        // Duplicate detection: mark-and-sweep via positionOf sentinel AFTER validation.
        // We validate all ids first (no state change), then remove them one-by-one
        // in the second pass. Duplicates surface on the second removal attempt as
        // `DuplicateInBatch` (line ~339) — the first pass's `AlreadyClaimed` check
        // catches ids already claimed by a *prior* transaction, not duplicates
        // within the current batch.
        for (uint256 i = 0; i < n; i++) {
            uint256 id = sourceTokenIds[i];
            if (id >= MILADY_SUPPLY) revert IdOutOfRange(id);
            if (_isClaimed(id)) revert AlreadyClaimed(id);
            address srcOwner = IERC721Minimal(miladyContract).ownerOf(id);
            if (srcOwner != msg.sender) revert NotMiladyOwner(id, msg.sender);
        }

        // Second pass: duplicate check via re-reading positionOf after each removal.
        for (uint256 i = 0; i < n; i++) {
            uint256 id = sourceTokenIds[i];
            if (_isClaimed(id)) revert DuplicateInBatch(id);
            uint256 pos = _positionOfId(id);
            uint256 removed = _removeFromPool(pos);
            // Defensive invariant: swap-and-pop must return the requested id.
            require(removed == id, "sparse-set invariant");
            _mintInternal(msg.sender, id, currentPh, true);
        }
    }

    // ---------------------------------------------------------------------
    // Phase 2/3: commit
    // ---------------------------------------------------------------------
    /// @notice Submit a blind commit for a random-draw mint. Escrows the phase price.
    /// @dev `commitHash = keccak256(abi.encode(secret, nonce, msg.sender))`.
    function commit(bytes32 commitHash) external payable whenFinalized {
        uint8 phase = _currentPhase();
        if (phase != 2 && phase != 3) revert WrongPhase(2, phase);

        uint256 price = phase == 2 ? PHASE_2_PRICE : PHASE_3_PRICE;
        if (msg.value != price) revert IncorrectPayment(msg.value, price);

        if (commits[commitHash].committer != address(0)) revert CommitExists(commitHash);

        if (phase == 2) {
            // N-4: CULT gate is commit-time only by design. Balance changes
            // between commit and reveal do not affect mint eligibility.
            uint256 bal = IERC20Minimal(cultToken).balanceOf(msg.sender);
            if (bal < CULT_THRESHOLD) revert CultGateNotMet(msg.sender, bal);
            // HIGH-1 (Pass 4): cap counts BOTH active commits and prior mints.
            // Closes the parallel multi-commit selection grinding attack —
            // an attacker cannot stage N simultaneous candidate commits to
            // observe outcomes and reveal only the most desirable subset.
            // Pass 5 gas-opt A: counters live in a packed `WalletCounters`
            // struct; the cap check reads both fields from the same slot.
            WalletCounters memory w = _walletCounters[msg.sender];
            if (uint256(w.phase2Active) + uint256(w.phase2Minted) >= PHASE_2_PER_WALLET) {
                revert WalletLimitExceeded(msg.sender, 2);
            }
            // SAFE: cap check above guarantees `w.phase2Active + 1 <=
            // PHASE_2_PER_WALLET` (10), well below uint64 max (1.8e19).
            // The promotion to uint64 storage is also safe since the cap
            // itself is enforced as uint256 before this addition.
            unchecked {
                _walletCounters[msg.sender].phase2Active = w.phase2Active + 1;
            }
        } else {
            WalletCounters memory w = _walletCounters[msg.sender];
            if (uint256(w.phase3Active) + uint256(w.phase3Minted) >= PHASE_3_PER_WALLET) {
                revert WalletLimitExceeded(msg.sender, 3);
            }
            // SAFE: same cap argument as the Phase 2 branch above.
            unchecked {
                _walletCounters[msg.sender].phase3Active = w.phase3Active + 1;
            }
        }

        commits[commitHash] = Commit({
            committer:   msg.sender,
            // forge-lint: disable-next-line(unsafe-typecast)
            price:       uint96(price),
            commitBlock: block.number,
            phase:       phase
        });
        // SAFE: `totalActiveCommitEscrow` accumulates `price` deposits
        // (max PHASE_3_PRICE = 0.02 ETH = 2e16 wei per commit) bounded
        // by the per-wallet cap × wallet count × phase price. Even with
        // every Ethereum address committing to the cap simultaneously,
        // the running total is far below uint256 max. Confirmed by the
        // existing `invariant_BalanceCoversEscrow` invariant in
        // Phantoma.invariants.t.sol — the contract balance covers the
        // escrow tally exactly, and contract balance is bounded by total
        // ETH supply.
        unchecked { totalActiveCommitEscrow += price; }

        emit CommitSubmitted(commitHash, msg.sender, phase, block.number, price);
    }

    /// @notice Batched-commit companion to `commit`. Records N pending
    ///         commits in a single transaction.
    /// @dev    Pass 18 (2026-05-21). Director-arbitrated trade-off: batching
    ///         commits forces all N to share `blockhash(commitBlock+1)` as
    ///         their entropy seed at reveal time, concentrating miner-
    ///         manipulation leverage and enabling single-block-observation
    ///         selection grinding. Bounded by:
    ///           - HIGH-1A `phase{2,3}Active + phase{2,3}Minted <= 10` per-
    ///             wallet cap (checked once over the full batch);
    ///           - HIGH-1B forfeit-to-treasury on `expireCommit` — each
    ///             discarded grind iteration costs `price`;
    ///           - `MAX_BATCH_COMMIT = 10` (matches the per-wallet cap, so
    ///             larger batches would revert `WalletLimitExceeded`
    ///             anyway).
    ///         Drainage protection preserved by construction: `msg.value`
    ///         strictly equals `price * n`, `totalActiveCommitEscrow`
    ///         increments by exactly the same, per-wallet cap enforced
    ///         pre-write, hash uniqueness validated against existing map
    ///         AND against in-batch duplicates.
    ///
    ///         Yul stack-depth discipline (Pass 13 deferred finding):
    ///         outer function is intentionally thin (max ~8 locals at
    ///         peak); compiles under default optimizer settings.
    /// @param  commitHashes Array of `keccak256(abi.encode(secret_i,
    ///                      nonce_i, msg.sender))` per element. Caller
    ///                      MUST ensure msg.sender is consistent across
    ///                      the batch (only msg.sender can reveal these
    ///                      commits later).
    function commitMany(bytes32[] calldata commitHashes) external payable whenFinalized {
        uint8 phase = _currentPhase();
        if (phase != 2 && phase != 3) revert WrongPhase(2, phase);

        uint256 n = commitHashes.length;
        if (n == 0) revert EmptyBatch();
        if (n > MAX_BATCH_COMMIT) revert BatchTooLarge(n);

        uint256 price = phase == 2 ? PHASE_2_PRICE : PHASE_3_PRICE;
        uint256 required = price * n;
        if (msg.value != required) revert IncorrectPayment(msg.value, required);

        // Phase 2 CULT gate (commit-time only — same as single commit()).
        if (phase == 2) {
            uint256 bal = IERC20Minimal(cultToken).balanceOf(msg.sender);
            if (bal < CULT_THRESHOLD) revert CultGateNotMet(msg.sender, bal);
        }

        // Per-wallet cap check (HIGH-1A): single read + cap-validate-against-n
        // BEFORE any state mutation. Same packed-slot read pattern as
        // single commit(). Per-phase constant lookup matches existing
        // commit()'s pattern (line 755 Phase 2, line 767 Phase 3) — using
        // the correct constant per phase guards against any future
        // constant-divergence between PHASE_2_PER_WALLET and
        // PHASE_3_PER_WALLET.
        WalletCounters memory w = _walletCounters[msg.sender];
        {
            uint256 currentTotal;
            uint256 cap;
            if (phase == 2) {
                currentTotal = uint256(w.phase2Active) + uint256(w.phase2Minted);
                cap = PHASE_2_PER_WALLET;
            } else {
                currentTotal = uint256(w.phase3Active) + uint256(w.phase3Minted);
                cap = PHASE_3_PER_WALLET;
            }
            if (currentTotal + n > cap) {
                revert WalletLimitExceeded(msg.sender, phase);
            }
        }

        // Hash validation: existing-map uniqueness + in-batch duplicate
        // detection. O(n²) inner loop bounded by MAX_BATCH_COMMIT = 10.
        // All validation completes BEFORE any state mutation so a single
        // bad input atomic-reverts the whole batch with the canonical
        // error rather than half-recording commits.
        for (uint256 i = 0; i < n; i++) {
            if (commits[commitHashes[i]].committer != address(0)) {
                revert CommitExists(commitHashes[i]);
            }
            for (uint256 j = i + 1; j < n; j++) {
                if (commitHashes[i] == commitHashes[j]) {
                    revert DuplicateCommitInBatch(commitHashes[i]);
                }
            }
        }

        // State writes: N commits + N events.
        for (uint256 i = 0; i < n; i++) {
            commits[commitHashes[i]] = Commit({
                committer:   msg.sender,
                // forge-lint: disable-next-line(unsafe-typecast)
                price:       uint96(price),
                commitBlock: block.number,
                phase:       phase
            });
            emit CommitSubmitted(commitHashes[i], msg.sender, phase, block.number, price);
        }

        // Counter + escrow update — single SSTORE each, after all writes
        // complete. SAFE for the unchecked: cap check above guarantees
        // `phaseXActive + n <= PHASE_X_PER_WALLET (10)` ≪ uint64 max
        // (1.8e19). `totalActiveCommitEscrow + required` bounded by
        // (per-wallet cap × max ETH supply) ≪ uint256 max — confirmed by
        // `invariant_BalanceCoversEscrow` (Phantoma.invariants.t.sol).
        unchecked {
            if (phase == 2) {
                _walletCounters[msg.sender].phase2Active = w.phase2Active + uint64(n);
            } else {
                _walletCounters[msg.sender].phase3Active = w.phase3Active + uint64(n);
            }
            totalActiveCommitEscrow += required;
        }
    }

    // ---------------------------------------------------------------------
    // Phase 2/3: reveal
    // ---------------------------------------------------------------------
    /// @notice Reveal a prior commit and mint a randomly-drawn token.
    /// @dev Window: [commitBlock+REVEAL_WINDOW_MIN, commitBlock+REVEAL_WINDOW_MAX]
    ///      inclusive. Entropy derives ONLY from `(secret, nonce, blockhash(
    ///      commitBlock+1))` — no `block.prevrandao` (deliberately removed so
    ///      reveal outcome is invariant across the reveal window, eliminating
    ///      reveal-block-selection grinding). Body extracted to `_revealOne`
    ///      at Pass 18 (2026-05-21) so `revealMany` can reuse the per-reveal
    ///      logic byte-for-byte without divergence.
    function reveal(bytes32 secret, uint256 nonce) external whenFinalized {
        _revealOne(secret, nonce);
    }

    /// @notice Batched-reveal companion to `reveal`. Settles N pending
    ///         commits in a single transaction by looping `_revealOne`.
    /// @dev    Pass 18 (2026-05-21). Each reveal uses ITS OWN pinned
    ///         entropy seed (committed at its own `commitBlock`); batching
    ///         introduces ZERO entropy concentration (unlike `commitMany`).
    ///         Atomic-fail semantics: if any one element reverts (commit
    ///         not found, window violation, pool exhausted, etc.), the
    ///         entire `revealMany` call reverts and all state mutations
    ///         roll back — user's commits preserved for retry.
    ///
    ///         Yul stack-depth discipline (Pass 13 deferred finding):
    ///         outer function is intentionally thin — only loop variables
    ///         live here. All deep-locals stay in `_revealOne`. Compiles
    ///         under default optimizer settings (`via_ir = false`).
    /// @param  secrets Array of per-commit secrets.
    /// @param  nonces  Array of per-commit nonces. Must be same length as
    ///                 `secrets` (parallel-array layout).
    function revealMany(bytes32[] calldata secrets, uint256[] calldata nonces)
        external
        whenFinalized
    {
        uint256 n = secrets.length;
        if (n != nonces.length) revert ArrayLengthMismatch(n, nonces.length);
        if (n == 0) revert EmptyBatch();
        if (n > MAX_BATCH_REVEAL) revert BatchTooLarge(n);

        for (uint256 i = 0; i < n; i++) {
            _revealOne(secrets[i], nonces[i]);
        }
    }

    /// @dev Internal per-reveal logic extracted from `reveal` at Pass 18.
    ///      Called once by `reveal(secret, nonce)` and N times by
    ///      `revealMany(secrets, nonces)`. Body is byte-for-byte identical
    ///      to the pre-Pass-18 `reveal` body — extraction is a pure
    ///      refactor, no semantic change. `msg.sender` is preserved across
    ///      the internal call so `commitHash` derivation remains anchored
    ///      to the externally-calling EOA, matching pre-Pass-18 behavior
    ///      for the single-reveal path AND extending naturally to
    ///      `revealMany` where the externally-calling EOA is the only
    ///      address that can have committed any of the N hashes.
    function _revealOne(bytes32 secret, uint256 nonce) internal {
        bytes32 commitHash = keccak256(abi.encode(secret, nonce, msg.sender));
        Commit memory c = commits[commitHash];
        if (c.committer == address(0)) revert CommitNotFound(commitHash);

        // 1. Expiry check FIRST (before blockhash access)
        uint256 latest = c.commitBlock + REVEAL_WINDOW_MAX;
        uint256 earliest = c.commitBlock + REVEAL_WINDOW_MIN;
        if (block.number > latest) revert RevealTooLate(block.number, latest);
        if (block.number < earliest) revert RevealTooEarly(block.number, earliest);

        // 1b. Phase boundary check — reject if phase advanced since commit
        uint8 currentPh = _currentPhase();
        if (currentPh > c.phase) revert WrongPhase(c.phase, currentPh);

        // 1c. Per-wallet limit check — defense in depth.
        //     Originally added per F-010 to close the commit-grinding mint-
        //     count bypass (Pass 1). With the HIGH-1 (Pass 4) commit-time
        //     active-slot cap, `phaseXMinted` cannot exceed PHASE_X_PER_WALLET
        //     in normal operation because every successful reveal consumes an
        //     active slot that was bounded at commit time. This check is
        //     retained as a redundant guarantee against future regressions of
        //     the commit-time cap.
        //     Pass 5 gas-opt A: read directly from packed slot (single
        //     SLOAD also warms the slot for the later counter update at
        //     step 6).
        if (c.phase == 2) {
            if (uint256(_walletCounters[c.committer].phase2Minted) >= PHASE_2_PER_WALLET) {
                revert WalletLimitExceeded(c.committer, 2);
            }
        } else {
            if (uint256(_walletCounters[c.committer].phase3Minted) >= PHASE_3_PER_WALLET) {
                revert WalletLimitExceeded(c.committer, 3);
            }
        }

        // 2. Blockhash access — fixed offset, not reveal-block-dependent.
        //    Pins randomness to commitBlock + REVEAL_WINDOW_MIN. The blockhash
        //    is unknown at commit time but fixed thereafter. Outcome is invariant
        //    across the reveal window, eliminating the grinding attack class.
        // Entropy source: blockhash of commitBlock + 1. Unknown at commit time,
        // fixed once that block is mined. Readable from commitBlock + 2 onward
        // (which is REVEAL_WINDOW_MIN). No prevrandao — outcome is invariant
        // across the reveal window, eliminating the grinding attack class.
        bytes32 bh = blockhash(c.commitBlock + 1);
        if (bh == bytes32(0)) revert BlockhashUnavailable(c.commitBlock + 1);

        // 3. Entropy — no block.prevrandao, only committed blockhash + secret
        uint256 seed = uint256(keccak256(abi.encode(secret, nonce, bh)));

        // 4. Draw
        if (unclaimedCount == 0) revert PoolEmpty();
        uint256 pos = seed % unclaimedCount;
        uint256 tokenId = _removeFromPool(pos);

        // 5. Effects before mint (which is internal _mint, no external call)
        delete commits[commitHash];
        // SAFE: `totalActiveCommitEscrow` was incremented by exactly
        // `c.price` when the commit was recorded (step 5 of `commit`).
        // Reaching this line proves the commit existed at price `c.price`,
        // so the running tally is strictly >= that amount. The subtraction
        // cannot underflow.
        unchecked { totalActiveCommitEscrow -= uint256(c.price); }

        // 6. Per-wallet counters — convert one active slot into a minted slot.
        //    HIGH-1 (Pass 4): the active decrement keeps `active+minted`
        //    invariant across the reveal so the commit-time cap continues to
        //    hold the same total budget after the conversion.
        //    Pass 5 gas-opt A: both updates land in a single SSTORE because
        //    `phaseXActive` and `phaseXMinted` share a packed slot via
        //    `WalletCounters`. Memory-copy + assign is the load-bearing
        //    pattern; in-place field updates would emit two separate
        //    writes.
        //
        //    SAFE for the unchecked arithmetic on the counters:
        //      - `phaseXMinted += 1`: bounded by the reveal-time
        //        defense-in-depth check above (step 1c). minted < cap (10).
        //      - `phaseXActive -= 1`: this committer's commit was just
        //        confirmed via the `commits[commitHash]` lookup, which
        //        only succeeds if the commit was recorded by `commit`.
        //        `commit` increments `phaseXActive` before recording the
        //        commit hash, so `phaseXActive >= 1` at this point.
        //      Both values stay in [0, 10] for the lifetime of any
        //      wallet's mint cycle — far below uint64 max.
        WalletCounters memory w = _walletCounters[c.committer];
        unchecked {
            if (c.phase == 2) {
                w.phase2Minted += 1;
                w.phase2Active -= 1;
            } else {
                w.phase3Minted += 1;
                w.phase3Active -= 1;
            }
        }
        _walletCounters[c.committer] = w;

        // 7. Mint — `isBijective=false` because commit-reveal is the
        //    random-draw path, not a Milady-ownership-gated bijective claim.
        //    Tier 0/1 phase gates and cosmetic-tier eligibility are computed
        //    by `mintFounderPack` against the actual phase the commit was
        //    submitted in (`c.phase`); bijective bonus does not apply.
        _mintInternal(c.committer, tokenId, c.phase, false);
        emit CommitRevealed(commitHash, c.committer, tokenId, seed, mintCount);
    }

    // ---------------------------------------------------------------------
    // Expired commit closure (permissionless, forfeit-to-treasury)
    // ---------------------------------------------------------------------
    /// @notice Close an expired commit. Callable by anyone.
    /// @dev    Per HIGH-1B (Pass 4 audit), an expired commit's escrow is
    ///         forfeited to the protocol rather than refunded to the
    ///         committer. The escrow remains in the contract balance and
    ///         becomes sweepable to treasury via the existing `sweep()`
    ///         reserve mechanism (`sweep` reserves only
    ///         `totalActiveCommitEscrow`; once this function decrements that
    ///         reserve, the forfeited price is part of the swept amount on
    ///         the next `sweep()` call).
    ///
    ///         This eliminates the parallel-and-sequential multi-commit
    ///         selection grinding economics: a non-revealed commit costs the
    ///         committer `PHASE_X_PRICE` non-recoverable, so any "commit N
    ///         candidates, reveal best K, walk away from N - K" strategy
    ///         pays `(N - K) * price` per round. With heavy-tailed rarity in
    ///         a 10,000-supply collection, this exceeds reasonable rarity
    ///         differential value at all selection ratios. See the cost
    ///         table in `test/Phantoma.HIGH1B_forfeiture.t.sol`.
    ///
    ///         F8 hostile-receiver self-DoS is preserved by construction —
    ///         there is no external transfer in this function, so there is
    ///         no failure path that could leave escrow inconsistent with the
    ///         persisted commit record. CEI is trivial: all state mutations,
    ///         no calls.
    ///
    ///         Permissionless caller is safe: `expireCommit` only succeeds
    ///         at `block.number > commitBlock + REVEAL_WINDOW_MAX`, which is
    ///         strictly greater than the last valid reveal block
    ///         (`commitBlock + REVEAL_WINDOW_MAX` itself, inclusive). A
    ///         third-party expirer cannot preempt a legitimate reveal.
    function expireCommit(bytes32 commitHash) external {
        Commit memory c = commits[commitHash];
        if (c.committer == address(0)) revert CommitNotFound(commitHash);

        uint256 expiresAt = c.commitBlock + REVEAL_WINDOW_MAX;
        if (block.number <= expiresAt) revert CommitNotExpired(block.number, expiresAt);

        uint256 price = uint256(c.price);
        address committer = c.committer;
        uint8 cphase = c.phase;

        delete commits[commitHash];
        // SAFE: `totalActiveCommitEscrow` was incremented by exactly
        // `price` when this commit was recorded. The commit-existence
        // check above (`c.committer == address(0)` revert) confirms the
        // commit is in the mapping, which proves the escrow tally
        // includes its `price`. Underflow impossible.
        unchecked { totalActiveCommitEscrow -= price; }
        // Release the active slot. No external transfer follows, so there is
        // no rollback path; the decrement is always coherent with the commit
        // deletion above. Pass 5 gas-opt A: counters live in a packed slot
        // via `WalletCounters`; single-field decrement still hits the
        // packed slot once.
        //
        // SAFE for the active decrement: if this commit exists in the
        // `commits` mapping (verified above), then `commit` was called
        // with the same `committer` and incremented `phaseXActive` to at
        // least 1. The decrement here cannot underflow.
        unchecked {
            if (cphase == 2) {
                _walletCounters[committer].phase2Active -= 1;
            } else {
                _walletCounters[committer].phase3Active -= 1;
            }
        }

        emit CommitExpired(commitHash, committer, price);
    }

    // ---------------------------------------------------------------------
    // Treasury sweep
    // ---------------------------------------------------------------------
    /// @notice Sweep all non-reserved balance to treasury.
    /// @dev Reserve = active commit escrow. Reverts if balance < reserve.
    function sweep() external onlyTreasury {
        uint256 bal = address(this).balance;
        uint256 reserve = totalActiveCommitEscrow;
        if (bal < reserve) revert SweepUnderflow(bal, reserve);
        uint256 amount = bal - reserve;
        (bool ok, ) = treasury.call{value: amount}("");
        if (!ok) revert SweepFailed(treasury, amount);
        emit Swept(treasury, amount);
    }

    // ---------------------------------------------------------------------
    // Internal mint
    // ---------------------------------------------------------------------
    /// @dev Pass 9 ordering (`_safeMint` last), built on Pass 4 OB-2 (Path B)
    ///      and Pass 6 spec-amendment combined equip+lock op (ERC-8216 PR #1645):
    ///        1. mintCount += 1
    ///        2. createAccount(...)            — deploy TBA (Option B)
    ///        3. mintFounderPack(tba, ...)     — cosmetics to TBA, not user
    ///        4. equipAndLockAtMint            — binding slot (always)
    ///        5. equipAndLockAtMint            — accessory slot (Medallion only)
    ///        6. resolvedAt[tokenId] = ts      — start URI grace window
    ///        7. _safeMint(to, tokenId)        — orphan-prevention (MEDIUM-1)
    ///        8. emit PhantomaMinted
    ///
    ///      The Pass 9 ordering closes the pre-equip approval-bypass attack
    ///      class. The prior ordering placed `_safeMint` at step 2, which
    ///      created a window during the recipient's `onERC721Received`
    ///      callback where the TBA had zero occupied slots. A contract
    ///      recipient could call `tba.execute(cosmetics, setApprovalForAll(
    ///      attacker, true), 0)` during that window — S-2's prefix loop
    ///      iterates currently-occupied slots, finds none, and lets the
    ///      approval through. The attacker drains the TBA's cosmetics after
    ///      the mint completes via the persistent approval; revocation also
    ///      requires `tba.execute(cosmetics, ...)` which is then blocked by
    ///      S-2, so the approval is permanent. This broke S-2 character-
    ///      binding for any contract recipient.
    ///
    ///      Pinning `_safeMint` last means the binding slot (and accessory
    ///      slot for Medallion-tier mints) is already locked when the
    ///      recipient's callback fires. The S-2 prefix loop on
    ///      `tba.execute(cosmetics, ...)` now finds the binding slot's
    ///      `tokenContract == cosmetics` and reverts
    ///      `ExecuteIntoEquippedContract`. The recipient's callback
    ///      propagates the revert, the entire mint atomically rolls back —
    ///      attackers cannot grant a pre-lock approval.
    ///
    ///      Uses `_safeMint` rather than `_mint` so contract recipients must
    ///      advertise `IERC721Receiver` (MEDIUM-1, Pass 4). Combined with the
    ///      ERC-1155 receiver requirement implicit in `mintFounderPack`'s
    ///      `_mintBatch` call, this makes the recipient-interface contract
    ///      symmetric across both standards. The orphan-prevention check
    ///      still fires under the new ordering — `_safeMint`'s position in
    ///      the sequence does not affect whether the receiver-interface
    ///      gate runs.
    ///
    ///      The TBA deploy is idempotent — the canonical 6551 registry
    ///      returns the existing account address if already deployed at the
    ///      deterministic CREATE2 location. If the registry call reverts
    ///      (e.g., the impl rejects initialization, or the registry is
    ///      misconfigured), the entire mint reverts atomically and the
    ///      sparse-set state rolls back along with mintCount.
    ///
    ///      The binding slot's anchor is read from the `hubKeyId`
    ///      immutable, which the constructor cached from
    ///      `tierItemIdsAt(cosmeticsCatchAllTier)[0]`. The PhantomaCosmetics
    ///      constructor enforces `CatchAllNotLastTier`, so this index always
    ///      points at the catch-all tier whose `maxPhase == 3` and threshold
    ///      covers the full Phantoma supply — meaning the binding anchor
    ///      item (Hub Key on mainnet) is guaranteed to be in the founder
    ///      pack for ANY (mintSequence, phase) pair. The binding equip+lock
    ///      therefore cannot fail under a valid config.
    ///
    ///      The Medallion equip is conditional on
    ///      `balanceOf(tba, medallionId) > 0` — `medallionId` is similarly
    ///      cached from `tierItemIdsAt(0)[0]`. The mainnet config restricts
    ///      the Medallion to first 500 Phase 1 mints (~5% of supply). For
    ///      the other ~9,500 mints the balance probe returns 0 and the
    ///      equip+lock pair is skipped — the accessory slot stays empty
    ///      for those TBAs and is available for user-driven
    ///      `equipFromBalance`. The `tierItemIdsAt(0)` lookup relies on
    ///      the ascending-thresholds invariant in PhantomaCosmetics, which
    ///      puts the rarest tier at index 0.
    ///
    ///      Soft-skip semantics on `mintFounderPack`: if a (mintSequence,
    ///      phase) somehow falls through every tier (impossible under the
    ///      catch-all invariant but defended against in the cosmetics
    ///      contract via `FounderPackSkippedEmpty`), the TBA holds zero
    ///      Hub Keys and the binding `equipAtMint` reverts with
    ///      `InsufficientTBABalance`, propagating the failure to the entire
    ///      mint. This is intentional — fail loudly on misconfiguration
    ///      rather than silently anchor S-2 for some mints and not others.
    ///
    ///      Reentrancy posture (Pass 9 ordering):
    ///        1. `mintCount` is incremented before any external call fires,
    ///           so a reentrant `_mintInternal` sees the post-increment
    ///           counter and allocates the next sequence cleanly — no
    ///           double-counting.
    ///        2. The sparse-set `_isClaimed` check and `_removeFromPool`
    ///           swap-and-pop run before `_mintInternal` is invoked (claim
    ///           path) or as the first effect of the draw (reveal path).
    ///           A reentrant attempt to mint the same `tokenId` reverts
    ///           via `DuplicateInBatch` (claim) or via the
    ///           `require(removed == id, "sparse-set invariant")` defense
    ///           (any path). Double-mint is impossible at any reentry
    ///           depth.
    ///        3. `mintFounderPack` triggers an ERC-1155 batch-receive
    ///           callback on the TBA (which inherits `ERC1155Holder` and
    ///           returns the magic selector). Re-entry from the TBA's
    ///           callback into Phantoma is bounded — Phantoma's external
    ///           entry points all enforce ownership / phase / payment
    ///           checks, and the TBA itself owns no Milady, has no CULT,
    ///           and would fail those gates.
    ///        4. `_safeMint` fires the recipient's `onERC721Received` only
    ///           AFTER the binding (and accessory if applicable) slots are
    ///           locked. The recipient sees a fully-configured TBA — they
    ///           cannot bypass S-2 by granting pre-lock approvals on the
    ///           cosmetics contract because S-2 already blocks that
    ///           contract from `execute()`. Re-entrant claim/commit/reveal
    ///           from the recipient's callback is governed by the same
    ///           sparse-set + payment + per-wallet-cap checks as a fresh
    ///           call.
    ///        5. `resolvedAt[tokenId]` is set before `_safeMint`. A
    ///           reentrant read during the recipient's callback sees the
    ///           genuine timestamp, not the documented "URI not yet set"
    ///           sentinel of 0. Pipeline operator's `setTokenURI` /
    ///           `finalizeToken` are still gated by `tokenFinalized` and
    ///           `tokenURISet` and grace-window math regardless.
    function _mintInternal(address to, uint256 tokenId, uint8 phase, bool isBijective) internal {
        // SAFE: mintCount is bounded by MILADY_SUPPLY (10000). Phase 1 is
        // bijective with the source-collection token-id space [0, 9999],
        // and Phase 2/3 draw from the same pool — total mints across all
        // phases cannot exceed 10,000. uint256 cannot overflow at this
        // scale; the unchecked saves ~30 gas per mint × ~10K mints.
        unchecked { mintCount += 1; }
        uint256 currentMintSequence = mintCount;

        // Deploy the TBA (idempotent on the canonical 6551 registry).
        address tba = IERC6551Registry(tbaRegistry).createAccount(
            equippableAccountImpl,
            bytes32(0),
            block.chainid,
            address(this),
            tokenId
        );

        // Mint the founder pack to the TBA (changed from `to` per Path B).
        // `isBijective` plumbed through so `mintFounderPack` can grant bonus
        // cosmetics (tiers with `tierBijectiveBonus[t] == true`) when this
        // is a `claim` call. Tier 0 cannot be elevated — that invariant
        // lives in PhantomaCosmetics's constructor.
        IPhantomaCosmetics(cosmeticsContract).mintFounderPack(tba, currentMintSequence, phase, isBijective);

        // Anchor S-2 with the catch-all binding-anchor item (Hub Key on
        // mainnet, Director-locked 2026-05-01) in the dedicated binding
        // slot. Item ID is cached as `hubKeyId` immutable at construction
        // (tier index N-1 is enforced as the catch-all by the cosmetics
        // constructor; balance is guaranteed >= 1 since catch-all mints
        // exactly 1 Hub Key — universal across all phases and sequences).
        // Combined equip+lock (ERC-8216 amendment) collapses two parent->TBA
        // CALLs into one and skips redundant occupancy/lock SLOAD checks in
        // the inline lock step.
        IERC6551Equipment(tba).equipAndLockAtMint(BINDING_SLOT, cosmeticsContract, hubKeyId, 1);

        // Conditional Medallion equip+lock: only Medallion-tier mints
        // qualify (first 500 Phase 1 in the mainnet config). Item ID is
        // cached as `medallionId` immutable; the probe gracefully no-ops
        // for ~9,500 non-Medallion mints, leaving the accessory slot
        // available for user `equipFromBalance`.
        if (IPhantomaCosmetics(cosmeticsContract).balanceOf(tba, medallionId) > 0) {
            IERC6551Equipment(tba).equipAndLockAtMint(ACCESSORY_SLOT, cosmeticsContract, medallionId, 1);
        }

        resolvedAt[tokenId] = block.timestamp;

        // Pass 9 reorder: `_safeMint` runs LAST so the recipient's
        // `onERC721Received` callback fires AFTER the binding (and
        // accessory) slot is locked. S-2's prefix loop in
        // `tba.execute()` will reject any attempt to call into the
        // cosmetics contract from the callback — closing the pre-equip
        // approval-bypass attack documented in the Pass 9 audit.
        // Orphan-prevention (MEDIUM-1) is preserved: any contract
        // recipient that does not implement `IERC721Receiver` reverts
        // the mint here, just as in the prior ordering.
        _safeMint(to, tokenId);

        emit PhantomaMinted(to, tokenId, currentMintSequence, phase);

        // ERC-4906 marketplace-refresh signal on mint (Pass 16 / 2026-05-18).
        // Without this, OpenSea / Blur / Magic Eden depend on Transfer-event-
        // driven first-fetch which is inconsistent across marketplaces and
        // produces stale Milady-source fallback renders during the post-mint
        // / pre-setTokenURI window. Fires for both bijective claim() and
        // commit-reveal reveal() paths since both funnel through this
        // internal mint function.
        emit MetadataUpdate(tokenId);
    }

    // ---------------------------------------------------------------------
    // URI management
    // ---------------------------------------------------------------------
    /// @notice Sets the base URI for all tokens without per-token overrides.
    /// @dev Under Pure B, this is the composition service URL (e.g., "https://metadata.phantoma.io/token/").
    ///      tokenURI() concatenates baseURI + tokenId. Per-token overrides via setTokenURI take precedence.
    ///      One call replaces ~9,998 individual setTokenURI calls.
    function setBaseURI(string calldata newBaseURI) external onlyPipelineOperator {
        if (baseURILocked) revert BaseURIIsLocked();
        _phantomaBaseURI = newBaseURI;
        emit BaseURISet(newBaseURI);
        // ERC-4906: signal marketplace to refresh cached metadata across the
        // full Phantoma id range. Range is inclusive on both ends per spec;
        // marketplaces handle the unminted half gracefully (no-op for ids
        // without an active token).
        emit BatchMetadataUpdate(0, MILADY_SUPPLY - 1);
    }

    /// @notice Permanently locks the base URI. One-way — cannot be unlocked.
    /// @dev After locking, setBaseURI always reverts. Operator key becomes inert
    ///      for base URI operations, matching the post-grace security posture of per-token URIs.
    function lockBaseURI() external onlyPipelineOperator {
        if (baseURILocked) revert BaseURIIsLocked();
        if (bytes(_phantomaBaseURI).length == 0) revert TokenURINotSet(0);
        baseURILocked = true;
        emit BaseURILocked(_phantomaBaseURI);
    }

    /// @notice Pipeline operator sets or re-rolls a token's URI during the grace window.
    /// @dev First call (spin 0) emits TokenURIResolved and does not increment spinCount.
    ///      Subsequent calls increment spinCount up to MAX_SPINS.
    function setTokenURI(uint256 tokenId, string calldata newURI) external onlyPipelineOperator {
        if (_ownerOf(tokenId) == address(0)) revert ERC721NonexistentToken(tokenId);
        if (tokenFinalized[tokenId]) revert TokenAlreadyFinalized(tokenId);
        if (resolvedAt[tokenId] == 0) revert TokenURINotSet(tokenId);
        if (block.timestamp > resolvedAt[tokenId] + GRACE_PERIOD) revert GracePeriodExpired(tokenId);

        if (!tokenURISet[tokenId]) {
            tokenURISet[tokenId] = true;
            _tokenURIs[tokenId] = newURI;
            emit TokenURIResolved(tokenId, newURI, block.timestamp);
        } else {
            if (spinCount[tokenId] >= MAX_SPINS) revert SpinLimitExceeded(tokenId);
            string memory oldURI = _tokenURIs[tokenId];
            // SAFE: spinCount is bounded by MAX_SPINS (3) per the check
            // immediately above. uint8 max (255) is far above the cap.
            unchecked { spinCount[tokenId] += 1; }
            _tokenURIs[tokenId] = newURI;
            emit TokenURIUpdatedDuringGrace(
                tokenId,
                oldURI,
                newURI,
                spinCount[tokenId],
                block.timestamp
            );
        }
        // ERC-4906: signal marketplace to refresh cached metadata for this
        // tokenId. Emits AFTER the existing custom URI events so the latter
        // remain the canonical Phantoma-internal logs and ERC-4906 is the
        // marketplace-side trigger.
        emit MetadataUpdate(tokenId);
    }

    /// @notice Permanently lock a token's URI. One-way.
    function finalizeToken(uint256 tokenId) external onlyPipelineOperator {
        if (_ownerOf(tokenId) == address(0)) revert ERC721NonexistentToken(tokenId);
        if (tokenFinalized[tokenId]) revert TokenAlreadyFinalized(tokenId);
        if (!tokenURISet[tokenId]) revert TokenURINotSet(tokenId);
        tokenFinalized[tokenId] = true;
        emit TokenFinalized(tokenId, block.timestamp);
    }

    /// @notice Returns the token's URI with three-tier fallback.
    /// @dev Precedence: per-token override (via setTokenURI) > baseURI + tokenId > PLACEHOLDER_URI.
    ///      Under Pure B, setBaseURI is called once after deploy. Per-token overrides are reserved
    ///      for edge-case corrections within the grace window. PLACEHOLDER_URI covers the gap
    ///      between mint and setBaseURI call.
    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        if (_ownerOf(tokenId) == address(0)) revert ERC721NonexistentToken(tokenId);
        if (tokenURISet[tokenId]) return _tokenURIs[tokenId];
        if (bytes(_phantomaBaseURI).length > 0) return string.concat(_phantomaBaseURI, Strings.toString(tokenId));
        return PLACEHOLDER_URI;
    }

    /// @notice ERC-721 collection-level metadata URI for marketplace branding.
    /// @dev Mirrors `tokenURI`'s base-URI fallback pattern. Pre-`setBaseURI`
    ///      window: returns `PLACEHOLDER_URI` (gracefully degrades for
    ///      indexers crawling between deploy and Pipeline's `setBaseURI`
    ///      call). Post-`setBaseURI`: returns
    ///      `string.concat(_phantomaBaseURI, "contract")` — e.g.
    ///      `https://metadata.phantoma.io/token/contract` when the metadata
    ///      service is configured. Marketplaces (OpenSea / Blur / Magic
    ///      Eden) auto-discover this getter at indexing time, removing the
    ///      manual admin-claim step at T+0 that would otherwise leave the
    ///      collection page unbranded for hours-to-days post-broadcast.
    ///
    ///      View only. No setter. Mutability lives in (a) the metadata
    ///      service URL target — CNAME flips, route updates by Pipeline
    ///      operate server-side without contract changes — and (b)
    ///      `_phantomaBaseURI` until `lockBaseURI()` finalizes it. After
    ///      `lockBaseURI()`, the contract URI is permanently
    ///      `<lockedBaseURI>contract`.
    function contractURI() external view returns (string memory) {
        if (bytes(_phantomaBaseURI).length > 0) {
            return string.concat(_phantomaBaseURI, "contract");
        }
        return PLACEHOLDER_URI;
    }

    /// @notice ERC-173 owner() — returns the Treasury Safe address.
    /// @dev    Marketplace collection-claim eligibility query. OpenSea + similar
    ///         marketplaces use ERC-173 `owner()` to determine who can claim
    ///         and edit the collection page. Returning `treasury` aligns with
    ///         the contract's economic-owner semantic (Treasury already
    ///         receives ERC-2981 royalty distributions + sweep proceeds).
    ///         Read-only — no setter, no `transferOwnership`. Mutability of
    ///         the underlying control surface lives in the Treasury Safe's
    ///         signer set, not in the contract.
    ///
    ///         Distinct from `launcher` (immutable role for `finalize()`) and
    ///         `pipelineOperator` (mutable role for `setBaseURI` /
    ///         `setTokenURI` / etc.). Treasury is the economic-owner anchor;
    ///         OpenSea queries `owner()` for storefront-edit auth, which maps
    ///         cleanly to "who controls collection-level commerce decisions"
    ///         — i.e. Treasury.
    ///
    ///         Interface ID `0x7f5828d0` advertised via `supportsInterface`.
    function owner() external view returns (address) {
        return treasury;
    }

    // ---------------------------------------------------------------------
    // TBA address view (lazy, never deploys)
    // ---------------------------------------------------------------------
    /// @notice Deterministic TBA address for a given token. Does not deploy.
    function tbaAddressOf(uint256 tokenId) external view returns (address) {
        return _tbaAddressOf(tokenId);
    }

    function _tbaAddressOf(uint256 tokenId) internal view returns (address) {
        return IERC6551Registry(tbaRegistry).account(
            equippableAccountImpl,
            bytes32(0),
            block.chainid,
            address(this),
            tokenId
        );
    }

    // ---------------------------------------------------------------------
    // Cosmetic transfer (in-ecosystem TBA→TBA gift path; foundation for
    // the marketplace purchase path layered on top in Phantoma's
    // marketplace MVP).
    // ---------------------------------------------------------------------

    /// @notice Transfer a cosmetic from one Phantoma's TBA to another
    ///         Phantoma's TBA, as a free gift. Caller must own the source
    ///         Phantoma. Cosmetics never leave the Phantoma ecosystem via
    ///         this path — destination is always a Phantoma-derived TBA.
    /// @dev    Free-balance check enforced via `_executeTransfer`. Locked
    ///         items (Hub Key in binding; Medallion in accessory for
    ///         Medallion-tier mints) and currently-equipped non-locked
    ///         items revert until unequipped or until the surplus above
    ///         the slot reservation is the transfer amount.
    function transferCosmeticBetweenTBAs(
        uint256 fromTokenId,
        uint256 toTokenId,
        uint256 cosmeticId,
        uint256 amount
    ) external {
        if (_ownerOf(fromTokenId) != msg.sender) {
            revert NotPhantomaOwner(fromTokenId, msg.sender);
        }
        _executeTransfer(fromTokenId, toTokenId, cosmeticId, amount);
        emit CosmeticTransferred(fromTokenId, toTokenId, cosmeticId, amount, msg.sender);
    }

    /// @notice Free-balance view helper for frontends.
    /// @dev    Returns `balanceOf(TBA, cosmeticId) - max(slot.amount across
    ///         slots claiming this cosmetic on TBA)`, clamped to 0. The
    ///         "transferable amount" surfaced by this function is the
    ///         exact upper bound that `transferCosmeticBetweenTBAs` and
    ///         (forthcoming) `listCosmetic` will accept without reverting
    ///         `CosmeticReservedInSlot`.
    function freeCosmeticBalance(uint256 tokenId, uint256 cosmeticId)
        external
        view
        returns (uint256)
    {
        if (_ownerOf(tokenId) == address(0)) revert ERC721NonexistentToken(tokenId);
        address tba = _tbaAddressOf(tokenId);
        uint256 balance = IPhantomaCosmetics(cosmeticsContract).balanceOf(tba, cosmeticId);
        uint256 maxReservation = IEquippableAccountExtended(tba)
            .maxSlotReservation(cosmeticsContract, cosmeticId);
        return balance > maxReservation ? balance - maxReservation : 0;
    }

    // ---------------------------------------------------------------------
    // Marketplace — list / cancel (Layer 2)
    // ---------------------------------------------------------------------

    /// @notice Create a fixed-price listing for a cosmetic held in
    ///         `sellerTokenId`'s TBA. Caller must own the Phantoma.
    /// @dev    List-time free-balance check is ADVISORY: at listing
    ///         creation we verify the cosmetic is currently transferable.
    ///         Between list-time and purchase-time the seller could equip
    ///         the cosmetic to a slot (raising `maxSlotReservation`); the
    ///         purchase-time check in `_executeTransfer` will then revert
    ///         `CosmeticReservedInSlot` and the buyer's transaction
    ///         atomically rolls back. Buyer's funds are EVM-revert-protected.
    ///
    ///         Listing inheritance: if the Phantoma is sold to a new owner
    ///         while the listing is active, the new owner inherits both the
    ///         right-to-cancel and the right-to-receive-proceeds. This is
    ///         documented behavior (see CLAUDE.md marketplace section).
    ///         Director arbitrated 2026-05-06: the front-running edge case
    ///         (lister sells Phantoma cheap to accomplice; accomplice
    ///         cancels) is accept-and-document — visible on-chain pattern,
    ///         requires intentional collusion, cost-to-attacker exceeds
    ///         rational economic incentive in most cases.
    /// @return listingId The unique identifier assigned to this listing.
    ///         Buyers reference it in `purchaseCosmetic`; the lister or
    ///         current Phantoma owner references it in `cancelListing`.
    function listCosmetic(
        uint256 sellerTokenId,
        uint256 cosmeticId,
        uint256 amount,
        uint96  price,
        uint64  durationSeconds
    ) external returns (uint256 listingId) {
        if (_ownerOf(sellerTokenId) != msg.sender) {
            revert NotPhantomaOwner(sellerTokenId, msg.sender);
        }
        if (amount == 0) revert InvalidAmount();
        if (price == 0) revert InvalidPrice();
        if (durationSeconds < MIN_LISTING_DURATION) revert ListingDurationTooShort(durationSeconds);
        if (durationSeconds > MAX_LISTING_DURATION) revert ListingDurationTooLong(durationSeconds);

        // List-time advisory free-balance check — catches "tried to list a
        // Hub Key" or "tried to list more than the surplus over equipped
        // slot" early at the lister's expense, with a clear error rather
        // than silently allowing a listing that always reverts on purchase.
        address srcTBA = _tbaAddressOf(sellerTokenId);
        uint256 balance = IPhantomaCosmetics(cosmeticsContract).balanceOf(srcTBA, cosmeticId);
        if (balance < amount) {
            revert InsufficientCosmeticBalance(srcTBA, cosmeticId, amount, balance);
        }
        uint256 maxReservation = IEquippableAccountExtended(srcTBA)
            .maxSlotReservation(cosmeticsContract, cosmeticId);
        if (balance - amount < maxReservation) {
            revert CosmeticReservedInSlot(srcTBA, cosmeticId, amount, balance, maxReservation);
        }

        // SAFE: listingCounter is monotonic, bounded by total realistic
        // marketplace activity. uint256 cannot overflow at any plausible
        // scale (would require ~1.16e77 listings).
        unchecked { listingCounter += 1; }
        listingId = listingCounter;

        // SAFE: block.timestamp + 30 days fits uint64 until ~year 20450.
        uint64 expiration = uint64(block.timestamp) + durationSeconds;
        listings[listingId] = Listing({
            sellerTokenId: sellerTokenId,
            cosmeticId:    cosmeticId,
            amount:        amount,
            price:         price,
            expiration:    expiration
        });

        emit CosmeticListed(listingId, sellerTokenId, cosmeticId, amount, price, expiration, msg.sender);
    }

    /// @notice Cancel an active marketplace listing. Caller must be the
    ///         current owner of `listing.sellerTokenId` (NOT necessarily
    ///         the original lister — if the Phantoma was sold while the
    ///         listing was active, the new owner has cancellation
    ///         authority).
    /// @dev    Free for caller (gas only, no penalty). No cooldown.
    ///         Refunds ~15k gas via the `delete` storage refund.
    function cancelListing(uint256 listingId) external {
        Listing memory l = listings[listingId];
        if (l.amount == 0) revert ListingNotFound(listingId);
        if (_ownerOf(l.sellerTokenId) != msg.sender) {
            revert NotPhantomaOwner(l.sellerTokenId, msg.sender);
        }
        delete listings[listingId];
        emit ListingCancelled(listingId, msg.sender);
    }

    // ---------------------------------------------------------------------
    // Marketplace — atomic purchase (Layer 3)
    // ---------------------------------------------------------------------

    /// @notice Atomically purchase a listed cosmetic. Caller must own the
    ///         buyer Phantoma. Pays exactly `listing.price` ETH; receives
    ///         the cosmetic in `buyerTokenId`'s TBA. Royalty (5% via
    ///         ERC-2981 single source of truth on PhantomaCosmetics)
    ///         routes to Treasury; remaining proceeds route to the CURRENT
    ///         owner of `listing.sellerTokenId` (listing-inheritance
    ///         economic correctness).
    /// @dev    CEI ordering — checks (auth + listing validity + price +
    ///         buyer auth + non-self) → effects (delete listing) →
    ///         interactions (cosmetic transfer via _executeTransfer, then
    ///         push payments to royalty receiver and seller payee).
    ///
    ///         `nonReentrant` modifier (transient-storage variant from OZ
    ///         v5.1) is belt-and-suspenders. The CEI ordering above is
    ///         already robust against re-entrancy in the analyzed call
    ///         graph, but the modifier costs ~100 gas (TSTORE/TLOAD) and
    ///         removes any future ambiguity if the call graph changes.
    ///
    ///         Free-balance is RE-VALIDATED at purchase time inside
    ///         `_executeTransfer`. If the seller equipped the cosmetic
    ///         between list-time and purchase-time (raising
    ///         `maxSlotReservation`), the entire purchase reverts
    ///         `CosmeticReservedInSlot` and the buyer's ETH is
    ///         EVM-revert-protected.
    ///
    ///         Push-payment failure handling: if either the royalty
    ///         receiver or the seller payee is a contract that reverts
    ///         in receive/fallback, the entire purchase atomically
    ///         reverts via `PaymentFailed`. The cosmetic transfer is
    ///         rolled back together with the listing-delete and the
    ///         ETH transfers — no partial state.
    ///
    ///         Defensive royalty cap: if a hypothetically-malicious
    ///         cosmetics override returned `royaltyAmount > msg.value`,
    ///         the cap clamps to msg.value (treasury gets everything,
    ///         seller gets 0). PhantomaCosmetics is the audited contract
    ///         with a fixed 5%; this branch is defense-in-depth only.
    function purchaseCosmetic(uint256 listingId, uint256 buyerTokenId)
        external
        payable
        nonReentrant
    {
        Listing memory l = listings[listingId];

        // ── Checks ──
        if (l.amount == 0) revert ListingNotFound(listingId);
        if (block.timestamp > l.expiration) revert ListingExpired(listingId, l.expiration);
        if (msg.value != l.price) revert IncorrectListingPrice(listingId, msg.value, l.price);
        if (_ownerOf(buyerTokenId) != msg.sender) revert NotPhantomaOwner(buyerTokenId, msg.sender);
        if (l.sellerTokenId == buyerTokenId) revert SelfPurchase();

        // Resolve seller payee — current owner at purchase time. Listing
        // inheritance: if Phantoma was sold while listing was active, the
        // new owner receives proceeds.
        address sellerPayee = _ownerOf(l.sellerTokenId);
        // Defensive: source Phantoma's owner is never zero in normal
        // operation (listing-creation auth would have failed otherwise),
        // but check guards against unanticipated edge cases.
        if (sellerPayee == address(0)) revert ERC721NonexistentToken(l.sellerTokenId);

        // Royalty math via ERC-2981 — single source of truth on
        // PhantomaCosmetics (5% to Treasury per `_setDefaultRoyalty` in the
        // cosmetics constructor).
        (address royaltyReceiver, uint256 royaltyAmount) =
            IERC2981(cosmeticsContract).royaltyInfo(l.cosmeticId, msg.value);
        // Defensive cap (see NatSpec above).
        if (royaltyAmount > msg.value) royaltyAmount = msg.value;
        uint256 sellerProceeds = msg.value - royaltyAmount;

        // Cache fields before deleting the listing storage.
        uint256 sellerTokenIdLocal = l.sellerTokenId;
        uint256 cosmeticIdLocal    = l.cosmeticId;
        uint256 amountLocal        = l.amount;

        // ── Effects ──
        delete listings[listingId];

        // Cosmetic transfer (re-validates free balance at purchase time —
        // protects against equip-between-list-and-purchase edge cases).
        _executeTransfer(sellerTokenIdLocal, buyerTokenId, cosmeticIdLocal, amountLocal);

        // ── Interactions ──
        if (royaltyAmount > 0) {
            (bool royaltyOk, ) = royaltyReceiver.call{value: royaltyAmount}("");
            if (!royaltyOk) revert PaymentFailed(royaltyReceiver);
        }
        if (sellerProceeds > 0) {
            (bool sellerOk, ) = sellerPayee.call{value: sellerProceeds}("");
            if (!sellerOk) revert PaymentFailed(sellerPayee);
        }

        emit CosmeticPurchased(
            listingId,
            sellerTokenIdLocal,
            buyerTokenId,
            cosmeticIdLocal,
            amountLocal,
            msg.value,
            royaltyAmount
        );
    }

    /// @dev Internal helper used by `transferCosmeticBetweenTBAs` (free gift)
    ///      and (forthcoming) `purchaseCosmetic` (marketplace). Performs the
    ///      common validation chain: distinct tokens, non-zero amount,
    ///      destination Phantoma must exist, source TBA must hold enough
    ///      cosmetics post-transfer to satisfy the maximum single-slot
    ///      reservation (S-1 invariant preservation).
    ///
    ///      Caller-side responsibility: verify msg.sender is authorized for
    ///      the source-side action (gift caller = source Phantoma owner;
    ///      purchase = listing's recorded sellerTokenId implicitly authorized
    ///      via the listing-creation auth check).
    ///
    ///      Both source and destination TBAs are guaranteed deployed at mint
    ///      time per `_mintInternal`'s step 2 `createAccount`. The destination
    ///      Phantoma's existence is verified via `_ownerOf != address(0)` —
    ///      the `_safeMint` at the end of `_mintInternal` sets `_owners[id] = to`
    ///      transactionally, so any minted token has a non-zero owner.
    function _executeTransfer(
        uint256 fromTokenId,
        uint256 toTokenId,
        uint256 cosmeticId,
        uint256 amount
    ) internal {
        if (fromTokenId == toTokenId) revert SelfTransfer();
        if (amount == 0) revert InvalidAmount();
        if (_ownerOf(toTokenId) == address(0)) revert ERC721NonexistentToken(toTokenId);

        address srcTBA = _tbaAddressOf(fromTokenId);
        address dstTBA = _tbaAddressOf(toTokenId);

        uint256 srcBalance = IPhantomaCosmetics(cosmeticsContract).balanceOf(srcTBA, cosmeticId);
        if (srcBalance < amount) {
            revert InsufficientCosmeticBalance(srcTBA, cosmeticId, amount, srcBalance);
        }

        uint256 maxReservation = IEquippableAccountExtended(srcTBA)
            .maxSlotReservation(cosmeticsContract, cosmeticId);
        if (srcBalance - amount < maxReservation) {
            revert CosmeticReservedInSlot(
                srcTBA, cosmeticId, amount, srcBalance, maxReservation
            );
        }

        IPhantomaCosmetics(cosmeticsContract).protocolTransfer(srcTBA, dstTBA, cosmeticId, amount);

        // ERC-4906: signal marketplace to refresh cached metadata for both
        // affected Phantomas. Source loses a cosmetic; destination gains
        // one — composite image of both changes (per the metadata service's
        // composition pipeline). Marketplaces (OpenSea / Blur / Magic Eden)
        // auto-refresh on these events.
        emit MetadataUpdate(fromTokenId);
        emit MetadataUpdate(toTokenId);
    }

    // ---------------------------------------------------------------------
    // Sparse-set helpers (Fix A — +1 offset)
    // ---------------------------------------------------------------------
    /// @dev positionOf encoding:
    ///      0                      => unset (virtual position == tokenId)
    ///      type(uint256).max      => claimed sentinel
    ///      N in (0, max)          => actual position == N - 1
    function _idAtPosition(uint256 pos) internal view returns (uint256) {
        uint256 stored = unclaimedIds[pos];
        return stored == 0 ? pos : stored;
    }

    function _positionOfId(uint256 tokenId) internal view returns (uint256) {
        uint256 stored = positionOf[tokenId];
        return stored == 0 ? tokenId : stored - 1;
    }

    function _isClaimed(uint256 tokenId) internal view returns (bool) {
        return positionOf[tokenId] == type(uint256).max;
    }

    /// @dev Unified swap-and-pop. Returns the tokenId removed from the pool.
    ///      Used by Phase 1 direct removal and Phase 2/3 random draw alike.
    function _removeFromPool(uint256 targetPosition) internal returns (uint256 claimedId) {
        claimedId = _idAtPosition(targetPosition);
        // SAFE: callers (claim's pass 2 and reveal) guarantee
        // `unclaimedCount > 0` before invoking this function — claim
        // checks via `_isClaimed`, reveal explicitly checks
        // `if (unclaimedCount == 0) revert PoolEmpty()`. Both subtractions
        // and the `targetPosition + 1` are therefore bounded above by
        // MILADY_SUPPLY (10000) and below by 0. uint256 cannot overflow.
        unchecked {
            uint256 tailIndex = unclaimedCount - 1;
            if (targetPosition != tailIndex) {
                uint256 tailId = _idAtPosition(tailIndex);
                unclaimedIds[targetPosition] = tailId;
                positionOf[tailId] = targetPosition + 1;
            }
            positionOf[claimedId] = type(uint256).max;
            unclaimedCount -= 1;
        }
    }

    // ---------------------------------------------------------------------
    // ERC-165
    // ---------------------------------------------------------------------
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC2981)
        returns (bool)
    {
        // EIP-4906 interface ID is custom-defined per spec:
        //   bytes4(keccak256('MetadataUpdate(uint256)')) ^
        //   bytes4(keccak256('BatchMetadataUpdate(uint256,uint256)'))
        // = 0x49064906. NOT computed via Solidity's `type(I).interfaceId`
        // since events don't contribute to that derivation. Hardcode per spec.
        if (interfaceId == bytes4(0x49064906)) return true;
        // ERC-173 (Contract Ownership Standard) interface ID = 0x7f5828d0.
        // Advertises `owner()` query support to OpenSea + marketplaces using
        // the standard collection-claim eligibility detection flow.
        if (interfaceId == bytes4(0x7f5828d0)) return true;
        return super.supportsInterface(interfaceId);
    }

    // ---------------------------------------------------------------------
    // ERC-4906 TBA-driven metadata refresh callback
    // ---------------------------------------------------------------------

    /// @notice Marketplace metadata-refresh entry point for TBA-driven
    ///         state changes. Emits `MetadataUpdate(tokenId)` so listening
    ///         marketplaces auto-refresh the Phantoma's cached composite
    ///         image + attributes.
    /// @dev    Auth gate: `msg.sender` must equal the canonical TBA
    ///         address derived for `tokenId`. The CREATE2 derivation by
    ///         the canonical 6551 registry is deterministic and uncontrol-
    ///         lable by external callers — no spoof path exists. Only the
    ///         specific TBA bound to a given Phantoma can trigger that
    ///         Phantoma's MetadataUpdate event.
    ///
    ///         Called by `EquippableAccount`'s slot-state mutation paths
    ///         (`equip` / `unequip` / `lockSlot` / `equipBatch` /
    ///         `lockSlots` / `equipFromBalance`) via try-catch — non-
    ///         Phantoma parents (other ERC-8216 deployments using
    ///         EquippableAccount as impl) gracefully skip the callback.
    ///
    ///         Mint-path equip operations (`equipAtMint` /
    ///         `equipAndLockAtMint` / `lockSlotAtMint`) intentionally
    ///         do NOT call this — those run during `_mintInternal` before
    ///         `_safeMint` lands, so emitting MetadataUpdate for a not-
    ///         yet-existent tokenId is wasteful. Marketplaces discover
    ///         new mints via the standard `Transfer` event from
    ///         `_safeMint` and fetch metadata fresh; ERC-4906 is for
    ///         POST-mint metadata changes.
    ///
    ///         Pure event emission — no state mutation. No re-entrancy
    ///         surface from this callback.
    function emitMetadataUpdate(uint256 tokenId) external {
        if (msg.sender != _tbaAddressOf(tokenId)) {
            revert NotTokenBoundAccount(tokenId, msg.sender);
        }
        emit MetadataUpdate(tokenId);
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC721/ERC721.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC721/ERC721.sol)

pragma solidity ^0.8.20;

import {IERC721} from "./IERC721.sol";
import {IERC721Metadata} from "./extensions/IERC721Metadata.sol";
import {ERC721Utils} from "./utils/ERC721Utils.sol";
import {Context} from "../../utils/Context.sol";
import {Strings} from "../../utils/Strings.sol";
import {IERC165, ERC165} from "../../utils/introspection/ERC165.sol";
import {IERC721Errors} from "../../interfaces/draft-IERC6093.sol";

/**
 * @dev Implementation of https://eips.ethereum.org/EIPS/eip-721[ERC-721] Non-Fungible Token Standard, including
 * the Metadata extension, but not including the Enumerable extension, which is available separately as
 * {ERC721Enumerable}.
 */
abstract contract ERC721 is Context, ERC165, IERC721, IERC721Metadata, IERC721Errors {
    using Strings for uint256;

    // Token name
    string private _name;

    // Token symbol
    string private _symbol;

    mapping(uint256 tokenId => address) private _owners;

    mapping(address owner => uint256) private _balances;

    mapping(uint256 tokenId => address) private _tokenApprovals;

    mapping(address owner => mapping(address operator => bool)) private _operatorApprovals;

    /**
     * @dev Initializes the contract by setting a `name` and a `symbol` to the token collection.
     */
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override(ERC165, IERC165) returns (bool) {
        return
            interfaceId == type(IERC721).interfaceId ||
            interfaceId == type(IERC721Metadata).interfaceId ||
            super.supportsInterface(interfaceId);
    }

    /**
     * @dev See {IERC721-balanceOf}.
     */
    function balanceOf(address owner) public view virtual returns (uint256) {
        if (owner == address(0)) {
            revert ERC721InvalidOwner(address(0));
        }
        return _balances[owner];
    }

    /**
     * @dev See {IERC721-ownerOf}.
     */
    function ownerOf(uint256 tokenId) public view virtual returns (address) {
        return _requireOwned(tokenId);
    }

    /**
     * @dev See {IERC721Metadata-name}.
     */
    function name() public view virtual returns (string memory) {
        return _name;
    }

    /**
     * @dev See {IERC721Metadata-symbol}.
     */
    function symbol() public view virtual returns (string memory) {
        return _symbol;
    }

    /**
     * @dev See {IERC721Metadata-tokenURI}.
     */
    function tokenURI(uint256 tokenId) public view virtual returns (string memory) {
        _requireOwned(tokenId);

        string memory baseURI = _baseURI();
        return bytes(baseURI).length > 0 ? string.concat(baseURI, tokenId.toString()) : "";
    }

    /**
     * @dev Base URI for computing {tokenURI}. If set, the resulting URI for each
     * token will be the concatenation of the `baseURI` and the `tokenId`. Empty
     * by default, can be overridden in child contracts.
     */
    function _baseURI() internal view virtual returns (string memory) {
        return "";
    }

    /**
     * @dev See {IERC721-approve}.
     */
    function approve(address to, uint256 tokenId) public virtual {
        _approve(to, tokenId, _msgSender());
    }

    /**
     * @dev See {IERC721-getApproved}.
     */
    function getApproved(uint256 tokenId) public view virtual returns (address) {
        _requireOwned(tokenId);

        return _getApproved(tokenId);
    }

    /**
     * @dev See {IERC721-setApprovalForAll}.
     */
    function setApprovalForAll(address operator, bool approved) public virtual {
        _setApprovalForAll(_msgSender(), operator, approved);
    }

    /**
     * @dev See {IERC721-isApprovedForAll}.
     */
    function isApprovedForAll(address owner, address operator) public view virtual returns (bool) {
        return _operatorApprovals[owner][operator];
    }

    /**
     * @dev See {IERC721-transferFrom}.
     */
    function transferFrom(address from, address to, uint256 tokenId) public virtual {
        if (to == address(0)) {
            revert ERC721InvalidReceiver(address(0));
        }
        // Setting an "auth" arguments enables the `_isAuthorized` check which verifies that the token exists
        // (from != 0). Therefore, it is not needed to verify that the return value is not 0 here.
        address previousOwner = _update(to, tokenId, _msgSender());
        if (previousOwner != from) {
            revert ERC721IncorrectOwner(from, tokenId, previousOwner);
        }
    }

    /**
     * @dev See {IERC721-safeTransferFrom}.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId) public {
        safeTransferFrom(from, to, tokenId, "");
    }

    /**
     * @dev See {IERC721-safeTransferFrom}.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes memory data) public virtual {
        transferFrom(from, to, tokenId);
        ERC721Utils.checkOnERC721Received(_msgSender(), from, to, tokenId, data);
    }

    /**
     * @dev Returns the owner of the `tokenId`. Does NOT revert if token doesn't exist
     *
     * IMPORTANT: Any overrides to this function that add ownership of tokens not tracked by the
     * core ERC-721 logic MUST be matched with the use of {_increaseBalance} to keep balances
     * consistent with ownership. The invariant to preserve is that for any address `a` the value returned by
     * `balanceOf(a)` must be equal to the number of tokens such that `_ownerOf(tokenId)` is `a`.
     */
    function _ownerOf(uint256 tokenId) internal view virtual returns (address) {
        return _owners[tokenId];
    }

    /**
     * @dev Returns the approved address for `tokenId`. Returns 0 if `tokenId` is not minted.
     */
    function _getApproved(uint256 tokenId) internal view virtual returns (address) {
        return _tokenApprovals[tokenId];
    }

    /**
     * @dev Returns whether `spender` is allowed to manage `owner`'s tokens, or `tokenId` in
     * particular (ignoring whether it is owned by `owner`).
     *
     * WARNING: This function assumes that `owner` is the actual owner of `tokenId` and does not verify this
     * assumption.
     */
    function _isAuthorized(address owner, address spender, uint256 tokenId) internal view virtual returns (bool) {
        return
            spender != address(0) &&
            (owner == spender || isApprovedForAll(owner, spender) || _getApproved(tokenId) == spender);
    }

    /**
     * @dev Checks if `spender` can operate on `tokenId`, assuming the provided `owner` is the actual owner.
     * Reverts if:
     * - `spender` does not have approval from `owner` for `tokenId`.
     * - `spender` does not have approval to manage all of `owner`'s assets.
     *
     * WARNING: This function assumes that `owner` is the actual owner of `tokenId` and does not verify this
     * assumption.
     */
    function _checkAuthorized(address owner, address spender, uint256 tokenId) internal view virtual {
        if (!_isAuthorized(owner, spender, tokenId)) {
            if (owner == address(0)) {
                revert ERC721NonexistentToken(tokenId);
            } else {
                revert ERC721InsufficientApproval(spender, tokenId);
            }
        }
    }

    /**
     * @dev Unsafe write access to the balances, used by extensions that "mint" tokens using an {ownerOf} override.
     *
     * NOTE: the value is limited to type(uint128).max. This protect against _balance overflow. It is unrealistic that
     * a uint256 would ever overflow from increments when these increments are bounded to uint128 values.
     *
     * WARNING: Increasing an account's balance using this function tends to be paired with an override of the
     * {_ownerOf} function to resolve the ownership of the corresponding tokens so that balances and ownership
     * remain consistent with one another.
     */
    function _increaseBalance(address account, uint128 value) internal virtual {
        unchecked {
            _balances[account] += value;
        }
    }

    /**
     * @dev Transfers `tokenId` from its current owner to `to`, or alternatively mints (or burns) if the current owner
     * (or `to`) is the zero address. Returns the owner of the `tokenId` before the update.
     *
     * The `auth` argument is optional. If the value passed is non 0, then this function will check that
     * `auth` is either the owner of the token, or approved to operate on the token (by the owner).
     *
     * Emits a {Transfer} event.
     *
     * NOTE: If overriding this function in a way that tracks balances, see also {_increaseBalance}.
     */
    function _update(address to, uint256 tokenId, address auth) internal virtual returns (address) {
        address from = _ownerOf(tokenId);

        // Perform (optional) operator check
        if (auth != address(0)) {
            _checkAuthorized(from, auth, tokenId);
        }

        // Execute the update
        if (from != address(0)) {
            // Clear approval. No need to re-authorize or emit the Approval event
            _approve(address(0), tokenId, address(0), false);

            unchecked {
                _balances[from] -= 1;
            }
        }

        if (to != address(0)) {
            unchecked {
                _balances[to] += 1;
            }
        }

        _owners[tokenId] = to;

        emit Transfer(from, to, tokenId);

        return from;
    }

    /**
     * @dev Mints `tokenId` and transfers it to `to`.
     *
     * WARNING: Usage of this method is discouraged, use {_safeMint} whenever possible
     *
     * Requirements:
     *
     * - `tokenId` must not exist.
     * - `to` cannot be the zero address.
     *
     * Emits a {Transfer} event.
     */
    function _mint(address to, uint256 tokenId) internal {
        if (to == address(0)) {
            revert ERC721InvalidReceiver(address(0));
        }
        address previousOwner = _update(to, tokenId, address(0));
        if (previousOwner != address(0)) {
            revert ERC721InvalidSender(address(0));
        }
    }

    /**
     * @dev Mints `tokenId`, transfers it to `to` and checks for `to` acceptance.
     *
     * Requirements:
     *
     * - `tokenId` must not exist.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function _safeMint(address to, uint256 tokenId) internal {
        _safeMint(to, tokenId, "");
    }

    /**
     * @dev Same as {xref-ERC721-_safeMint-address-uint256-}[`_safeMint`], with an additional `data` parameter which is
     * forwarded in {IERC721Receiver-onERC721Received} to contract recipients.
     */
    function _safeMint(address to, uint256 tokenId, bytes memory data) internal virtual {
        _mint(to, tokenId);
        ERC721Utils.checkOnERC721Received(_msgSender(), address(0), to, tokenId, data);
    }

    /**
     * @dev Destroys `tokenId`.
     * The approval is cleared when the token is burned.
     * This is an internal function that does not check if the sender is authorized to operate on the token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     *
     * Emits a {Transfer} event.
     */
    function _burn(uint256 tokenId) internal {
        address previousOwner = _update(address(0), tokenId, address(0));
        if (previousOwner == address(0)) {
            revert ERC721NonexistentToken(tokenId);
        }
    }

    /**
     * @dev Transfers `tokenId` from `from` to `to`.
     *  As opposed to {transferFrom}, this imposes no restrictions on msg.sender.
     *
     * Requirements:
     *
     * - `to` cannot be the zero address.
     * - `tokenId` token must be owned by `from`.
     *
     * Emits a {Transfer} event.
     */
    function _transfer(address from, address to, uint256 tokenId) internal {
        if (to == address(0)) {
            revert ERC721InvalidReceiver(address(0));
        }
        address previousOwner = _update(to, tokenId, address(0));
        if (previousOwner == address(0)) {
            revert ERC721NonexistentToken(tokenId);
        } else if (previousOwner != from) {
            revert ERC721IncorrectOwner(from, tokenId, previousOwner);
        }
    }

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`, checking that contract recipients
     * are aware of the ERC-721 standard to prevent tokens from being forever locked.
     *
     * `data` is additional data, it has no specified format and it is sent in call to `to`.
     *
     * This internal function is like {safeTransferFrom} in the sense that it invokes
     * {IERC721Receiver-onERC721Received} on the receiver, and can be used to e.g.
     * implement alternative mechanisms to perform token transfer, such as signature-based.
     *
     * Requirements:
     *
     * - `tokenId` token must exist and be owned by `from`.
     * - `to` cannot be the zero address.
     * - `from` cannot be the zero address.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function _safeTransfer(address from, address to, uint256 tokenId) internal {
        _safeTransfer(from, to, tokenId, "");
    }

    /**
     * @dev Same as {xref-ERC721-_safeTransfer-address-address-uint256-}[`_safeTransfer`], with an additional `data` parameter which is
     * forwarded in {IERC721Receiver-onERC721Received} to contract recipients.
     */
    function _safeTransfer(address from, address to, uint256 tokenId, bytes memory data) internal virtual {
        _transfer(from, to, tokenId);
        ERC721Utils.checkOnERC721Received(_msgSender(), from, to, tokenId, data);
    }

    /**
     * @dev Approve `to` to operate on `tokenId`
     *
     * The `auth` argument is optional. If the value passed is non 0, then this function will check that `auth` is
     * either the owner of the token, or approved to operate on all tokens held by this owner.
     *
     * Emits an {Approval} event.
     *
     * Overrides to this logic should be done to the variant with an additional `bool emitEvent` argument.
     */
    function _approve(address to, uint256 tokenId, address auth) internal {
        _approve(to, tokenId, auth, true);
    }

    /**
     * @dev Variant of `_approve` with an optional flag to enable or disable the {Approval} event. The event is not
     * emitted in the context of transfers.
     */
    function _approve(address to, uint256 tokenId, address auth, bool emitEvent) internal virtual {
        // Avoid reading the owner unless necessary
        if (emitEvent || auth != address(0)) {
            address owner = _requireOwned(tokenId);

            // We do not use _isAuthorized because single-token approvals should not be able to call approve
            if (auth != address(0) && owner != auth && !isApprovedForAll(owner, auth)) {
                revert ERC721InvalidApprover(auth);
            }

            if (emitEvent) {
                emit Approval(owner, to, tokenId);
            }
        }

        _tokenApprovals[tokenId] = to;
    }

    /**
     * @dev Approve `operator` to operate on all of `owner` tokens
     *
     * Requirements:
     * - operator can't be the address zero.
     *
     * Emits an {ApprovalForAll} event.
     */
    function _setApprovalForAll(address owner, address operator, bool approved) internal virtual {
        if (operator == address(0)) {
            revert ERC721InvalidOperator(operator);
        }
        _operatorApprovals[owner][operator] = approved;
        emit ApprovalForAll(owner, operator, approved);
    }

    /**
     * @dev Reverts if the `tokenId` doesn't have a current owner (it hasn't been minted, or it has been burned).
     * Returns the owner.
     *
     * Overrides to ownership logic should be done to {_ownerOf}.
     */
    function _requireOwned(uint256 tokenId) internal view returns (address) {
        address owner = _ownerOf(tokenId);
        if (owner == address(0)) {
            revert ERC721NonexistentToken(tokenId);
        }
        return owner;
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/common/ERC2981.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/common/ERC2981.sol)

pragma solidity ^0.8.20;

import {IERC2981} from "../../interfaces/IERC2981.sol";
import {IERC165, ERC165} from "../../utils/introspection/ERC165.sol";

/**
 * @dev Implementation of the NFT Royalty Standard, a standardized way to retrieve royalty payment information.
 *
 * Royalty information can be specified globally for all token ids via {_setDefaultRoyalty}, and/or individually for
 * specific token ids via {_setTokenRoyalty}. The latter takes precedence over the first.
 *
 * Royalty is specified as a fraction of sale price. {_feeDenominator} is overridable but defaults to 10000, meaning the
 * fee is specified in basis points by default.
 *
 * IMPORTANT: ERC-2981 only specifies a way to signal royalty information and does not enforce its payment. See
 * https://eips.ethereum.org/EIPS/eip-2981#optional-royalty-payments[Rationale] in the ERC. Marketplaces are expected to
 * voluntarily pay royalties together with sales, but note that this standard is not yet widely supported.
 */
abstract contract ERC2981 is IERC2981, ERC165 {
    struct RoyaltyInfo {
        address receiver;
        uint96 royaltyFraction;
    }

    RoyaltyInfo private _defaultRoyaltyInfo;
    mapping(uint256 tokenId => RoyaltyInfo) private _tokenRoyaltyInfo;

    /**
     * @dev The default royalty set is invalid (eg. (numerator / denominator) >= 1).
     */
    error ERC2981InvalidDefaultRoyalty(uint256 numerator, uint256 denominator);

    /**
     * @dev The default royalty receiver is invalid.
     */
    error ERC2981InvalidDefaultRoyaltyReceiver(address receiver);

    /**
     * @dev The royalty set for an specific `tokenId` is invalid (eg. (numerator / denominator) >= 1).
     */
    error ERC2981InvalidTokenRoyalty(uint256 tokenId, uint256 numerator, uint256 denominator);

    /**
     * @dev The royalty receiver for `tokenId` is invalid.
     */
    error ERC2981InvalidTokenRoyaltyReceiver(uint256 tokenId, address receiver);

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override(IERC165, ERC165) returns (bool) {
        return interfaceId == type(IERC2981).interfaceId || super.supportsInterface(interfaceId);
    }

    /**
     * @inheritdoc IERC2981
     */
    function royaltyInfo(
        uint256 tokenId,
        uint256 salePrice
    ) public view virtual returns (address receiver, uint256 amount) {
        RoyaltyInfo storage _royaltyInfo = _tokenRoyaltyInfo[tokenId];
        address royaltyReceiver = _royaltyInfo.receiver;
        uint96 royaltyFraction = _royaltyInfo.royaltyFraction;

        if (royaltyReceiver == address(0)) {
            royaltyReceiver = _defaultRoyaltyInfo.receiver;
            royaltyFraction = _defaultRoyaltyInfo.royaltyFraction;
        }

        uint256 royaltyAmount = (salePrice * royaltyFraction) / _feeDenominator();

        return (royaltyReceiver, royaltyAmount);
    }

    /**
     * @dev The denominator with which to interpret the fee set in {_setTokenRoyalty} and {_setDefaultRoyalty} as a
     * fraction of the sale price. Defaults to 10000 so fees are expressed in basis points, but may be customized by an
     * override.
     */
    function _feeDenominator() internal pure virtual returns (uint96) {
        return 10000;
    }

    /**
     * @dev Sets the royalty information that all ids in this contract will default to.
     *
     * Requirements:
     *
     * - `receiver` cannot be the zero address.
     * - `feeNumerator` cannot be greater than the fee denominator.
     */
    function _setDefaultRoyalty(address receiver, uint96 feeNumerator) internal virtual {
        uint256 denominator = _feeDenominator();
        if (feeNumerator > denominator) {
            // Royalty fee will exceed the sale price
            revert ERC2981InvalidDefaultRoyalty(feeNumerator, denominator);
        }
        if (receiver == address(0)) {
            revert ERC2981InvalidDefaultRoyaltyReceiver(address(0));
        }

        _defaultRoyaltyInfo = RoyaltyInfo(receiver, feeNumerator);
    }

    /**
     * @dev Removes default royalty information.
     */
    function _deleteDefaultRoyalty() internal virtual {
        delete _defaultRoyaltyInfo;
    }

    /**
     * @dev Sets the royalty information for a specific token id, overriding the global default.
     *
     * Requirements:
     *
     * - `receiver` cannot be the zero address.
     * - `feeNumerator` cannot be greater than the fee denominator.
     */
    function _setTokenRoyalty(uint256 tokenId, address receiver, uint96 feeNumerator) internal virtual {
        uint256 denominator = _feeDenominator();
        if (feeNumerator > denominator) {
            // Royalty fee will exceed the sale price
            revert ERC2981InvalidTokenRoyalty(tokenId, feeNumerator, denominator);
        }
        if (receiver == address(0)) {
            revert ERC2981InvalidTokenRoyaltyReceiver(tokenId, address(0));
        }

        _tokenRoyaltyInfo[tokenId] = RoyaltyInfo(receiver, feeNumerator);
    }

    /**
     * @dev Resets royalty information for the token id back to the global default.
     */
    function _resetTokenRoyalty(uint256 tokenId) internal virtual {
        delete _tokenRoyaltyInfo[tokenId];
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/interfaces/IERC2981.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/IERC2981.sol)

pragma solidity ^0.8.20;

import {IERC165} from "../utils/introspection/IERC165.sol";

/**
 * @dev Interface for the NFT Royalty Standard.
 *
 * A standardized way to retrieve royalty payment information for non-fungible tokens (NFTs) to enable universal
 * support for royalty payments across all NFT marketplaces and ecosystem participants.
 */
interface IERC2981 is IERC165 {
    /**
     * @dev Returns how much royalty is owed and to whom, based on a sale price that may be denominated in any unit of
     * exchange. The royalty amount is denominated and should be paid in that same unit of exchange.
     *
     * NOTE: ERC-2981 allows setting the royalty to 100% of the price. In that case all the price would be sent to the
     * royalty receiver and 0 tokens to the seller. Contracts dealing with royalty should consider empty transfers.
     */
    function royaltyInfo(
        uint256 tokenId,
        uint256 salePrice
    ) external view returns (address receiver, uint256 royaltyAmount);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/ReentrancyGuardTransient.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/ReentrancyGuardTransient.sol)

pragma solidity ^0.8.24;

import {TransientSlot} from "./TransientSlot.sol";

/**
 * @dev Variant of {ReentrancyGuard} that uses transient storage.
 *
 * NOTE: This variant only works on networks where EIP-1153 is available.
 *
 * _Available since v5.1._
 */
abstract contract ReentrancyGuardTransient {
    using TransientSlot for *;

    // keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.ReentrancyGuard")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant REENTRANCY_GUARD_STORAGE =
        0x9b779b17422d0df92223018b32b4d1fa46e071723d6817e2486d003becc55f00;

    /**
     * @dev Unauthorized reentrant call.
     */
    error ReentrancyGuardReentrantCall();

    /**
     * @dev Prevents a contract from calling itself, directly or indirectly.
     * Calling a `nonReentrant` function from another `nonReentrant`
     * function is not supported. It is possible to prevent this from happening
     * by making the `nonReentrant` function external, and making it call a
     * `private` function that does the actual work.
     */
    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        // On the first call to nonReentrant, _status will be NOT_ENTERED
        if (_reentrancyGuardEntered()) {
            revert ReentrancyGuardReentrantCall();
        }

        // Any calls to nonReentrant after this point will fail
        REENTRANCY_GUARD_STORAGE.asBoolean().tstore(true);
    }

    function _nonReentrantAfter() private {
        REENTRANCY_GUARD_STORAGE.asBoolean().tstore(false);
    }

    /**
     * @dev Returns true if the reentrancy guard is currently set to "entered", which indicates there is a
     * `nonReentrant` function in the call stack.
     */
    function _reentrancyGuardEntered() internal view returns (bool) {
        return REENTRANCY_GUARD_STORAGE.asBoolean().tload();
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/Strings.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/Strings.sol)

pragma solidity ^0.8.20;

import {Math} from "./math/Math.sol";
import {SignedMath} from "./math/SignedMath.sol";

/**
 * @dev String operations.
 */
library Strings {
    bytes16 private constant HEX_DIGITS = "0123456789abcdef";
    uint8 private constant ADDRESS_LENGTH = 20;

    /**
     * @dev The `value` string doesn't fit in the specified `length`.
     */
    error StringsInsufficientHexLength(uint256 value, uint256 length);

    /**
     * @dev Converts a `uint256` to its ASCII `string` decimal representation.
     */
    function toString(uint256 value) internal pure returns (string memory) {
        unchecked {
            uint256 length = Math.log10(value) + 1;
            string memory buffer = new string(length);
            uint256 ptr;
            assembly ("memory-safe") {
                ptr := add(buffer, add(32, length))
            }
            while (true) {
                ptr--;
                assembly ("memory-safe") {
                    mstore8(ptr, byte(mod(value, 10), HEX_DIGITS))
                }
                value /= 10;
                if (value == 0) break;
            }
            return buffer;
        }
    }

    /**
     * @dev Converts a `int256` to its ASCII `string` decimal representation.
     */
    function toStringSigned(int256 value) internal pure returns (string memory) {
        return string.concat(value < 0 ? "-" : "", toString(SignedMath.abs(value)));
    }

    /**
     * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation.
     */
    function toHexString(uint256 value) internal pure returns (string memory) {
        unchecked {
            return toHexString(value, Math.log256(value) + 1);
        }
    }

    /**
     * @dev Converts a `uint256` to its ASCII `string` hexadecimal representation with fixed length.
     */
    function toHexString(uint256 value, uint256 length) internal pure returns (string memory) {
        uint256 localValue = value;
        bytes memory buffer = new bytes(2 * length + 2);
        buffer[0] = "0";
        buffer[1] = "x";
        for (uint256 i = 2 * length + 1; i > 1; --i) {
            buffer[i] = HEX_DIGITS[localValue & 0xf];
            localValue >>= 4;
        }
        if (localValue != 0) {
            revert StringsInsufficientHexLength(value, length);
        }
        return string(buffer);
    }

    /**
     * @dev Converts an `address` with fixed length of 20 bytes to its not checksummed ASCII `string` hexadecimal
     * representation.
     */
    function toHexString(address addr) internal pure returns (string memory) {
        return toHexString(uint256(uint160(addr)), ADDRESS_LENGTH);
    }

    /**
     * @dev Converts an `address` with fixed length of 20 bytes to its checksummed ASCII `string` hexadecimal
     * representation, according to EIP-55.
     */
    function toChecksumHexString(address addr) internal pure returns (string memory) {
        bytes memory buffer = bytes(toHexString(addr));

        // hash the hex part of buffer (skip length + 2 bytes, length 40)
        uint256 hashValue;
        assembly ("memory-safe") {
            hashValue := shr(96, keccak256(add(buffer, 0x22), 40))
        }

        for (uint256 i = 41; i > 1; --i) {
            // possible values for buffer[i] are 48 (0) to 57 (9) and 97 (a) to 102 (f)
            if (hashValue & 0xf > 7 && uint8(buffer[i]) > 96) {
                // case shift by xoring with 0x20
                buffer[i] ^= 0x20;
            }
            hashValue >>= 4;
        }
        return string(buffer);
    }

    /**
     * @dev Returns true if the two strings are equal.
     */
    function equal(string memory a, string memory b) internal pure returns (bool) {
        return bytes(a).length == bytes(b).length && keccak256(bytes(a)) == keccak256(bytes(b));
    }
}


// ===== FILE: src/interfaces/IERC6551.sol =====
// SPDX-License-Identifier: CC0-1.0
pragma solidity ^0.8.24;

interface IERC6551Registry {
    event ERC6551AccountCreated(
        address account,
        address indexed implementation,
        bytes32 salt,
        uint256 chainId,
        address indexed tokenContract,
        uint256 indexed tokenId
    );

    function createAccount(
        address implementation,
        bytes32 salt,
        uint256 chainId,
        address tokenContract,
        uint256 tokenId
    ) external returns (address account);

    function account(
        address implementation,
        bytes32 salt,
        uint256 chainId,
        address tokenContract,
        uint256 tokenId
    ) external view returns (address account);
}

interface IERC6551Account {
    receive() external payable;

    function token()
        external
        view
        returns (uint256 chainId, address tokenContract, uint256 tokenId);

    function state() external view returns (uint256);

    function isValidSigner(address signer, bytes calldata context)
        external
        view
        returns (bytes4 magicValue);
}

interface IERC6551Executable {
    function execute(address to, uint256 value, bytes calldata data, uint8 operation)
        external
        payable
        returns (bytes memory);
}


// ===== FILE: src/interfaces/IERC6551Equipment.sol =====
// SPDX-License-Identifier: CC0-1.0
pragma solidity ^0.8.24;

/// @title IERC6551Equipment — Slot-Based Equipment for Token Bound Accounts
/// @notice A standard interface for equipping, unequipping, and permanently
///         locking tokens within ERC-6551 Token Bound Accounts using named slots.
/// @dev    Slots are identified by bytes32 keys, allowing any application to
///         define its own slot taxonomy. The recommended convention is
///         keccak256("slot.<name>"). Applications sharing a TBA across contexts
///         SHOULD namespace slots to avoid collisions, e.g.
///         keccak256("myapp.slot.head") vs keccak256("otherapp.slot.head").
///
///         Slots may be permanently locked, making them immutable across
///         ownership transfers. Locked means locked forever — there is no
///         unlock mechanism by design.
///
///         The original ERC-165 identifier (`0xd38f0891`) covered the nine
///         function set defined in ERC-8216 PR #1645 (equip / unequip /
///         lockSlot / equipBatch / lockSlots / getEquipped / getLoadout /
///         isSlotOccupied / isSlotLocked). With Phantoma's Option B work
///         (Pass 4) the interface gains three additional functions
///         (equipAtMint / lockSlotAtMint / equipFromBalance), so the ERC-165
///         identifier is recomputed at compile time via
///         `type(IERC6551Equipment).interfaceId` — no hard-coded literal in
///         the production code path. The ERC-8216 spec amendment in PR #1645
///         will pin the new identifier alongside the additional function
///         signatures and the `isERC721` field on `SlotEntry`.

interface IERC6551Equipment {

    /// @notice Metadata describing an occupied equipment slot.
    /// @dev    `isERC721` is cached at equip time via ERC-165 probe, and is
    ///         immutable for the lifetime of the slot occupation.
    ///         Implementations MUST NOT re-probe the token contract's
    ///         interface after the slot is occupied. A token contract that
    ///         changes its ERC-165 response between equip and later
    ///         operations would otherwise permanently brick the TBA.
    struct SlotEntry {
        bytes32 slotId;
        address tokenContract;
        uint256 tokenId;
        uint256 amount;
        bool locked;
        bool isERC721;
    }

    event Equipped(
        bytes32 indexed slotId,
        address indexed tokenContract,
        uint256 indexed tokenId,
        uint256 amount
    );

    event Unequipped(
        bytes32 indexed slotId,
        address indexed tokenContract,
        uint256 indexed tokenId,
        uint256 amount
    );

    event SlotLocked(
        bytes32 indexed slotId,
        address indexed tokenContract,
        uint256 tokenId
    );

    function equip(bytes32 slotId, address tokenContract, uint256 tokenId, uint256 amount) external;

    function unequip(bytes32 slotId) external;

    function lockSlot(bytes32 slotId) external;

    function equipBatch(
        bytes32[] calldata slotIds,
        address[] calldata tokenContracts,
        uint256[] calldata tokenIds,
        uint256[] calldata amounts
    ) external;

    function lockSlots(bytes32[] calldata slotIds) external;

    function getEquipped(bytes32 slotId) external view returns (address tokenContract, uint256 tokenId, uint256 amount);

    function getLoadout() external view returns (SlotEntry[] memory entries);

    function isSlotOccupied(bytes32 slotId) external view returns (bool);

    function isSlotLocked(bytes32 slotId) external view returns (bool);

    /// @notice Register an item already held in the TBA balance into a slot.
    ///         Performs no transfer. Intended to be called by the parent
    ///         contract during the mint path, immediately after the founder
    ///         pack has been minted to the TBA.
    /// @dev    Implementations MUST restrict access to the parent contract
    ///         (the `tokenContract` returned by `token()`). Implementations
    ///         MUST verify the TBA holds at least `amount` of the asset
    ///         (`balanceOf(address(this), tokenId) >= amount` for ERC-1155;
    ///         `ownerOf(tokenId) == address(this)` for ERC-721) — the check
    ///         tolerates the same balance condition that
    ///         `_verifyEquipmentInvariant` uses post-execute.
    function equipAtMint(
        bytes32 slotId,
        address tokenContract,
        uint256 tokenId,
        uint256 amount
    ) external;

    /// @notice Lock a slot occupied by `equipAtMint`. Permanent.
    /// @dev    Implementations MUST restrict access to the parent contract.
    ///         Behaviorally identical to `lockSlot` with a different access
    ///         gate; provided as a separate function so the parent contract
    ///         does not need to be the NFT owner during the mint transaction.
    function lockSlotAtMint(bytes32 slotId) external;

    /// @notice Equip an item already held in the TBA balance into a slot.
    ///         Performs no transfer. Intended for owner-initiated equips
    ///         where the item entered the TBA via a path other than the
    ///         standard wallet-to-TBA `equip` (e.g., a future cosmetic drop
    ///         minted directly to the TBA, or an item retained after
    ///         `unequip`).
    /// @dev    Implementations MUST restrict access to the NFT owner.
    ///         Same balance-condition tolerance as `equipAtMint`.
    function equipFromBalance(
        bytes32 slotId,
        address tokenContract,
        uint256 tokenId,
        uint256 amount
    ) external;

    /// @notice Atomically register an item already held in the TBA balance
    ///         into a slot AND lock the slot in a single call. Performs no
    ///         transfer. Intended for parent-contract use during the mint
    ///         path when the slot is meant to be permanent from inception
    ///         (e.g., a character-binding anchor written at mint time).
    /// @dev    Implementations MUST restrict access to the parent contract.
    ///         Implementations MUST emit BOTH the `Equipped` event and the
    ///         `SlotLocked` event so indexers tracking either independently
    ///         do not need to special-case the combined function. The
    ///         observable post-state MUST be identical to calling
    ///         `equipAtMint` followed by `lockSlotAtMint` with the same
    ///         arguments. Same balance-condition tolerance as `equipAtMint`.
    function equipAndLockAtMint(
        bytes32 slotId,
        address tokenContract,
        uint256 tokenId,
        uint256 amount
    ) external;
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC721/IERC721.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC721/IERC721.sol)

pragma solidity ^0.8.20;

import {IERC165} from "../../utils/introspection/IERC165.sol";

/**
 * @dev Required interface of an ERC-721 compliant contract.
 */
interface IERC721 is IERC165 {
    /**
     * @dev Emitted when `tokenId` token is transferred from `from` to `to`.
     */
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);

    /**
     * @dev Emitted when `owner` enables `approved` to manage the `tokenId` token.
     */
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);

    /**
     * @dev Emitted when `owner` enables or disables (`approved`) `operator` to manage all of its assets.
     */
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

    /**
     * @dev Returns the number of tokens in ``owner``'s account.
     */
    function balanceOf(address owner) external view returns (uint256 balance);

    /**
     * @dev Returns the owner of the `tokenId` token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     */
    function ownerOf(uint256 tokenId) external view returns (address owner);

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon
     *   a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;

    /**
     * @dev Safely transfers `tokenId` token from `from` to `to`, checking first that contract recipients
     * are aware of the ERC-721 protocol to prevent tokens from being forever locked.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must exist and be owned by `from`.
     * - If the caller is not `from`, it must have been allowed to move this token by either {approve} or
     *   {setApprovalForAll}.
     * - If `to` refers to a smart contract, it must implement {IERC721Receiver-onERC721Received}, which is called upon
     *   a safe transfer.
     *
     * Emits a {Transfer} event.
     */
    function safeTransferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @dev Transfers `tokenId` token from `from` to `to`.
     *
     * WARNING: Note that the caller is responsible to confirm that the recipient is capable of receiving ERC-721
     * or else they may be permanently lost. Usage of {safeTransferFrom} prevents loss, though the caller must
     * understand this adds an external call which potentially creates a reentrancy vulnerability.
     *
     * Requirements:
     *
     * - `from` cannot be the zero address.
     * - `to` cannot be the zero address.
     * - `tokenId` token must be owned by `from`.
     * - If the caller is not `from`, it must be approved to move this token by either {approve} or {setApprovalForAll}.
     *
     * Emits a {Transfer} event.
     */
    function transferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @dev Gives permission to `to` to transfer `tokenId` token to another account.
     * The approval is cleared when the token is transferred.
     *
     * Only a single account can be approved at a time, so approving the zero address clears previous approvals.
     *
     * Requirements:
     *
     * - The caller must own the token or be an approved operator.
     * - `tokenId` must exist.
     *
     * Emits an {Approval} event.
     */
    function approve(address to, uint256 tokenId) external;

    /**
     * @dev Approve or remove `operator` as an operator for the caller.
     * Operators can call {transferFrom} or {safeTransferFrom} for any token owned by the caller.
     *
     * Requirements:
     *
     * - The `operator` cannot be the address zero.
     *
     * Emits an {ApprovalForAll} event.
     */
    function setApprovalForAll(address operator, bool approved) external;

    /**
     * @dev Returns the account approved for `tokenId` token.
     *
     * Requirements:
     *
     * - `tokenId` must exist.
     */
    function getApproved(uint256 tokenId) external view returns (address operator);

    /**
     * @dev Returns if the `operator` is allowed to manage all of the assets of `owner`.
     *
     * See {setApprovalForAll}
     */
    function isApprovedForAll(address owner, address operator) external view returns (bool);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC721/extensions/IERC721Metadata.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (token/ERC721/extensions/IERC721Metadata.sol)

pragma solidity ^0.8.20;

import {IERC721} from "../IERC721.sol";

/**
 * @title ERC-721 Non-Fungible Token Standard, optional metadata extension
 * @dev See https://eips.ethereum.org/EIPS/eip-721
 */
interface IERC721Metadata is IERC721 {
    /**
     * @dev Returns the token collection name.
     */
    function name() external view returns (string memory);

    /**
     * @dev Returns the token collection symbol.
     */
    function symbol() external view returns (string memory);

    /**
     * @dev Returns the Uniform Resource Identifier (URI) for `tokenId` token.
     */
    function tokenURI(uint256 tokenId) external view returns (string memory);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC721/utils/ERC721Utils.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC721/utils/ERC721Utils.sol)

pragma solidity ^0.8.20;

import {IERC721Receiver} from "../IERC721Receiver.sol";
import {IERC721Errors} from "../../../interfaces/draft-IERC6093.sol";

/**
 * @dev Library that provide common ERC-721 utility functions.
 *
 * See https://eips.ethereum.org/EIPS/eip-721[ERC-721].
 *
 * _Available since v5.1._
 */
library ERC721Utils {
    /**
     * @dev Performs an acceptance check for the provided `operator` by calling {IERC721-onERC721Received}
     * on the `to` address. The `operator` is generally the address that initiated the token transfer (i.e. `msg.sender`).
     *
     * The acceptance call is not executed and treated as a no-op if the target address doesn't contain code (i.e. an EOA).
     * Otherwise, the recipient must implement {IERC721Receiver-onERC721Received} and return the acceptance magic value to accept
     * the transfer.
     */
    function checkOnERC721Received(
        address operator,
        address from,
        address to,
        uint256 tokenId,
        bytes memory data
    ) internal {
        if (to.code.length > 0) {
            try IERC721Receiver(to).onERC721Received(operator, from, tokenId, data) returns (bytes4 retval) {
                if (retval != IERC721Receiver.onERC721Received.selector) {
                    // Token rejected
                    revert IERC721Errors.ERC721InvalidReceiver(to);
                }
            } catch (bytes memory reason) {
                if (reason.length == 0) {
                    // non-IERC721Receiver implementer
                    revert IERC721Errors.ERC721InvalidReceiver(to);
                } else {
                    assembly ("memory-safe") {
                        revert(add(32, reason), mload(reason))
                    }
                }
            }
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/Context.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

pragma solidity ^0.8.20;

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/introspection/ERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/introspection/ERC165.sol)

pragma solidity ^0.8.20;

import {IERC165} from "./IERC165.sol";

/**
 * @dev Implementation of the {IERC165} interface.
 *
 * Contracts that want to implement ERC-165 should inherit from this contract and override {supportsInterface} to check
 * for the additional interface id that will be supported. For example:
 *
 * ```solidity
 * function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {
 *     return interfaceId == type(MyInterface).interfaceId || super.supportsInterface(interfaceId);
 * }
 * ```
 */
abstract contract ERC165 is IERC165 {
    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) public view virtual returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/interfaces/draft-IERC6093.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (interfaces/draft-IERC6093.sol)
pragma solidity ^0.8.20;

/**
 * @dev Standard ERC-20 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-20 tokens.
 */
interface IERC20Errors {
    /**
     * @dev Indicates an error related to the current `balance` of a `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param balance Current balance for the interacting account.
     * @param needed Minimum amount required to perform a transfer.
     */
    error ERC20InsufficientBalance(address sender, uint256 balance, uint256 needed);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC20InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC20InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `spender`’s `allowance`. Used in transfers.
     * @param spender Address that may be allowed to operate on tokens without being their owner.
     * @param allowance Amount of tokens a `spender` is allowed to operate with.
     * @param needed Minimum amount required to perform a transfer.
     */
    error ERC20InsufficientAllowance(address spender, uint256 allowance, uint256 needed);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC20InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `spender` to be approved. Used in approvals.
     * @param spender Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC20InvalidSpender(address spender);
}

/**
 * @dev Standard ERC-721 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-721 tokens.
 */
interface IERC721Errors {
    /**
     * @dev Indicates that an address can't be an owner. For example, `address(0)` is a forbidden owner in ERC-20.
     * Used in balance queries.
     * @param owner Address of the current owner of a token.
     */
    error ERC721InvalidOwner(address owner);

    /**
     * @dev Indicates a `tokenId` whose `owner` is the zero address.
     * @param tokenId Identifier number of a token.
     */
    error ERC721NonexistentToken(uint256 tokenId);

    /**
     * @dev Indicates an error related to the ownership over a particular token. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param tokenId Identifier number of a token.
     * @param owner Address of the current owner of a token.
     */
    error ERC721IncorrectOwner(address sender, uint256 tokenId, address owner);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC721InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC721InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `operator`’s approval. Used in transfers.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     * @param tokenId Identifier number of a token.
     */
    error ERC721InsufficientApproval(address operator, uint256 tokenId);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC721InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `operator` to be approved. Used in approvals.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC721InvalidOperator(address operator);
}

/**
 * @dev Standard ERC-1155 Errors
 * Interface of the https://eips.ethereum.org/EIPS/eip-6093[ERC-6093] custom errors for ERC-1155 tokens.
 */
interface IERC1155Errors {
    /**
     * @dev Indicates an error related to the current `balance` of a `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     * @param balance Current balance for the interacting account.
     * @param needed Minimum amount required to perform a transfer.
     * @param tokenId Identifier number of a token.
     */
    error ERC1155InsufficientBalance(address sender, uint256 balance, uint256 needed, uint256 tokenId);

    /**
     * @dev Indicates a failure with the token `sender`. Used in transfers.
     * @param sender Address whose tokens are being transferred.
     */
    error ERC1155InvalidSender(address sender);

    /**
     * @dev Indicates a failure with the token `receiver`. Used in transfers.
     * @param receiver Address to which tokens are being transferred.
     */
    error ERC1155InvalidReceiver(address receiver);

    /**
     * @dev Indicates a failure with the `operator`’s approval. Used in transfers.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     * @param owner Address of the current owner of a token.
     */
    error ERC1155MissingApprovalForAll(address operator, address owner);

    /**
     * @dev Indicates a failure with the `approver` of a token to be approved. Used in approvals.
     * @param approver Address initiating an approval operation.
     */
    error ERC1155InvalidApprover(address approver);

    /**
     * @dev Indicates a failure with the `operator` to be approved. Used in approvals.
     * @param operator Address that may be allowed to operate on tokens without being their owner.
     */
    error ERC1155InvalidOperator(address operator);

    /**
     * @dev Indicates an array length mismatch between ids and values in a safeBatchTransferFrom operation.
     * Used in batch transfers.
     * @param idsLength Length of the array of token identifiers
     * @param valuesLength Length of the array of token amounts
     */
    error ERC1155InvalidArrayLength(uint256 idsLength, uint256 valuesLength);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/introspection/IERC165.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/introspection/IERC165.sol)

pragma solidity ^0.8.20;

/**
 * @dev Interface of the ERC-165 standard, as defined in the
 * https://eips.ethereum.org/EIPS/eip-165[ERC].
 *
 * Implementers can declare support of contract interfaces, which can then be
 * queried by others ({ERC165Checker}).
 *
 * For an implementation, see {ERC165}.
 */
interface IERC165 {
    /**
     * @dev Returns true if this contract implements the interface defined by
     * `interfaceId`. See the corresponding
     * https://eips.ethereum.org/EIPS/eip-165#how-interfaces-are-identified[ERC section]
     * to learn more about how these ids are created.
     *
     * This function call must use less than 30 000 gas.
     */
    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/TransientSlot.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/TransientSlot.sol)
// This file was procedurally generated from scripts/generate/templates/TransientSlot.js.

pragma solidity ^0.8.24;

/**
 * @dev Library for reading and writing value-types to specific transient storage slots.
 *
 * Transient slots are often used to store temporary values that are removed after the current transaction.
 * This library helps with reading and writing to such slots without the need for inline assembly.
 *
 *  * Example reading and writing values using transient storage:
 * ```solidity
 * contract Lock {
 *     using TransientSlot for *;
 *
 *     // Define the slot. Alternatively, use the SlotDerivation library to derive the slot.
 *     bytes32 internal constant _LOCK_SLOT = 0xf4678858b2b588224636b8522b729e7722d32fc491da849ed75b3fdf3c84f542;
 *
 *     modifier locked() {
 *         require(!_LOCK_SLOT.asBoolean().tload());
 *
 *         _LOCK_SLOT.asBoolean().tstore(true);
 *         _;
 *         _LOCK_SLOT.asBoolean().tstore(false);
 *     }
 * }
 * ```
 *
 * TIP: Consider using this library along with {SlotDerivation}.
 */
library TransientSlot {
    /**
     * @dev UDVT that represent a slot holding a address.
     */
    type AddressSlot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a AddressSlot.
     */
    function asAddress(bytes32 slot) internal pure returns (AddressSlot) {
        return AddressSlot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a bool.
     */
    type BooleanSlot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a BooleanSlot.
     */
    function asBoolean(bytes32 slot) internal pure returns (BooleanSlot) {
        return BooleanSlot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a bytes32.
     */
    type Bytes32Slot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a Bytes32Slot.
     */
    function asBytes32(bytes32 slot) internal pure returns (Bytes32Slot) {
        return Bytes32Slot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a uint256.
     */
    type Uint256Slot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a Uint256Slot.
     */
    function asUint256(bytes32 slot) internal pure returns (Uint256Slot) {
        return Uint256Slot.wrap(slot);
    }

    /**
     * @dev UDVT that represent a slot holding a int256.
     */
    type Int256Slot is bytes32;

    /**
     * @dev Cast an arbitrary slot to a Int256Slot.
     */
    function asInt256(bytes32 slot) internal pure returns (Int256Slot) {
        return Int256Slot.wrap(slot);
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(AddressSlot slot) internal view returns (address value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(AddressSlot slot, address value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(BooleanSlot slot) internal view returns (bool value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(BooleanSlot slot, bool value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(Bytes32Slot slot) internal view returns (bytes32 value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(Bytes32Slot slot, bytes32 value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(Uint256Slot slot) internal view returns (uint256 value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(Uint256Slot slot, uint256 value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }

    /**
     * @dev Load the value held at location `slot` in transient storage.
     */
    function tload(Int256Slot slot) internal view returns (int256 value) {
        assembly ("memory-safe") {
            value := tload(slot)
        }
    }

    /**
     * @dev Store `value` at location `slot` in transient storage.
     */
    function tstore(Int256Slot slot, int256 value) internal {
        assembly ("memory-safe") {
            tstore(slot, value)
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/math/Math.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/math/Math.sol)

pragma solidity ^0.8.20;

import {Panic} from "../Panic.sol";
import {SafeCast} from "./SafeCast.sol";

/**
 * @dev Standard math utilities missing in the Solidity language.
 */
library Math {
    enum Rounding {
        Floor, // Toward negative infinity
        Ceil, // Toward positive infinity
        Trunc, // Toward zero
        Expand // Away from zero
    }

    /**
     * @dev Returns the addition of two unsigned integers, with an success flag (no overflow).
     */
    function tryAdd(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            uint256 c = a + b;
            if (c < a) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the subtraction of two unsigned integers, with an success flag (no overflow).
     */
    function trySub(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            if (b > a) return (false, 0);
            return (true, a - b);
        }
    }

    /**
     * @dev Returns the multiplication of two unsigned integers, with an success flag (no overflow).
     */
    function tryMul(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
            // benefit is lost if 'b' is also tested.
            // See: https://github.com/OpenZeppelin/openzeppelin-contracts/pull/522
            if (a == 0) return (true, 0);
            uint256 c = a * b;
            if (c / a != b) return (false, 0);
            return (true, c);
        }
    }

    /**
     * @dev Returns the division of two unsigned integers, with a success flag (no division by zero).
     */
    function tryDiv(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a / b);
        }
    }

    /**
     * @dev Returns the remainder of dividing two unsigned integers, with a success flag (no division by zero).
     */
    function tryMod(uint256 a, uint256 b) internal pure returns (bool success, uint256 result) {
        unchecked {
            if (b == 0) return (false, 0);
            return (true, a % b);
        }
    }

    /**
     * @dev Branchless ternary evaluation for `a ? b : c`. Gas costs are constant.
     *
     * IMPORTANT: This function may reduce bytecode size and consume less gas when used standalone.
     * However, the compiler may optimize Solidity ternary operations (i.e. `a ? b : c`) to only compute
     * one branch when needed, making this function more expensive.
     */
    function ternary(bool condition, uint256 a, uint256 b) internal pure returns (uint256) {
        unchecked {
            // branchless ternary works because:
            // b ^ (a ^ b) == a
            // b ^ 0 == b
            return b ^ ((a ^ b) * SafeCast.toUint(condition));
        }
    }

    /**
     * @dev Returns the largest of two numbers.
     */
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return ternary(a > b, a, b);
    }

    /**
     * @dev Returns the smallest of two numbers.
     */
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return ternary(a < b, a, b);
    }

    /**
     * @dev Returns the average of two numbers. The result is rounded towards
     * zero.
     */
    function average(uint256 a, uint256 b) internal pure returns (uint256) {
        // (a + b) / 2 can overflow.
        return (a & b) + (a ^ b) / 2;
    }

    /**
     * @dev Returns the ceiling of the division of two numbers.
     *
     * This differs from standard division with `/` in that it rounds towards infinity instead
     * of rounding towards zero.
     */
    function ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        if (b == 0) {
            // Guarantee the same behavior as in a regular Solidity division.
            Panic.panic(Panic.DIVISION_BY_ZERO);
        }

        // The following calculation ensures accurate ceiling division without overflow.
        // Since a is non-zero, (a - 1) / b will not overflow.
        // The largest possible result occurs when (a - 1) / b is type(uint256).max,
        // but the largest value we can obtain is type(uint256).max - 1, which happens
        // when a = type(uint256).max and b = 1.
        unchecked {
            return SafeCast.toUint(a > 0) * ((a - 1) / b + 1);
        }
    }

    /**
     * @dev Calculates floor(x * y / denominator) with full precision. Throws if result overflows a uint256 or
     * denominator == 0.
     *
     * Original credit to Remco Bloemen under MIT license (https://xn--2-umb.com/21/muldiv) with further edits by
     * Uniswap Labs also under MIT license.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 result) {
        unchecked {
            // 512-bit multiply [prod1 prod0] = x * y. Compute the product mod 2²⁵⁶ and mod 2²⁵⁶ - 1, then use
            // the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
            // variables such that product = prod1 * 2²⁵⁶ + prod0.
            uint256 prod0 = x * y; // Least significant 256 bits of the product
            uint256 prod1; // Most significant 256 bits of the product
            assembly {
                let mm := mulmod(x, y, not(0))
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            // Handle non-overflow cases, 256 by 256 division.
            if (prod1 == 0) {
                // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                // The surrounding unchecked block does not change this fact.
                // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                return prod0 / denominator;
            }

            // Make sure the result is less than 2²⁵⁶. Also prevents denominator == 0.
            if (denominator <= prod1) {
                Panic.panic(ternary(denominator == 0, Panic.DIVISION_BY_ZERO, Panic.UNDER_OVERFLOW));
            }

            ///////////////////////////////////////////////
            // 512 by 256 division.
            ///////////////////////////////////////////////

            // Make division exact by subtracting the remainder from [prod1 prod0].
            uint256 remainder;
            assembly {
                // Compute remainder using mulmod.
                remainder := mulmod(x, y, denominator)

                // Subtract 256 bit number from 512 bit number.
                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            // Factor powers of two out of denominator and compute largest power of two divisor of denominator.
            // Always >= 1. See https://cs.stackexchange.com/q/138556/92363.

            uint256 twos = denominator & (0 - denominator);
            assembly {
                // Divide denominator by twos.
                denominator := div(denominator, twos)

                // Divide [prod1 prod0] by twos.
                prod0 := div(prod0, twos)

                // Flip twos such that it is 2²⁵⁶ / twos. If twos is zero, then it becomes one.
                twos := add(div(sub(0, twos), twos), 1)
            }

            // Shift in bits from prod1 into prod0.
            prod0 |= prod1 * twos;

            // Invert denominator mod 2²⁵⁶. Now that denominator is an odd number, it has an inverse modulo 2²⁵⁶ such
            // that denominator * inv ≡ 1 mod 2²⁵⁶. Compute the inverse by starting with a seed that is correct for
            // four bits. That is, denominator * inv ≡ 1 mod 2⁴.
            uint256 inverse = (3 * denominator) ^ 2;

            // Use the Newton-Raphson iteration to improve the precision. Thanks to Hensel's lifting lemma, this also
            // works in modular arithmetic, doubling the correct bits in each step.
            inverse *= 2 - denominator * inverse; // inverse mod 2⁸
            inverse *= 2 - denominator * inverse; // inverse mod 2¹⁶
            inverse *= 2 - denominator * inverse; // inverse mod 2³²
            inverse *= 2 - denominator * inverse; // inverse mod 2⁶⁴
            inverse *= 2 - denominator * inverse; // inverse mod 2¹²⁸
            inverse *= 2 - denominator * inverse; // inverse mod 2²⁵⁶

            // Because the division is now exact we can divide by multiplying with the modular inverse of denominator.
            // This will give us the correct result modulo 2²⁵⁶. Since the preconditions guarantee that the outcome is
            // less than 2²⁵⁶, this is the final result. We don't need to compute the high bits of the result and prod1
            // is no longer required.
            result = prod0 * inverse;
            return result;
        }
    }

    /**
     * @dev Calculates x * y / denominator with full precision, following the selected rounding direction.
     */
    function mulDiv(uint256 x, uint256 y, uint256 denominator, Rounding rounding) internal pure returns (uint256) {
        return mulDiv(x, y, denominator) + SafeCast.toUint(unsignedRoundsUp(rounding) && mulmod(x, y, denominator) > 0);
    }

    /**
     * @dev Calculate the modular multiplicative inverse of a number in Z/nZ.
     *
     * If n is a prime, then Z/nZ is a field. In that case all elements are inversible, except 0.
     * If n is not a prime, then Z/nZ is not a field, and some elements might not be inversible.
     *
     * If the input value is not inversible, 0 is returned.
     *
     * NOTE: If you know for sure that n is (big) a prime, it may be cheaper to use Fermat's little theorem and get the
     * inverse using `Math.modExp(a, n - 2, n)`. See {invModPrime}.
     */
    function invMod(uint256 a, uint256 n) internal pure returns (uint256) {
        unchecked {
            if (n == 0) return 0;

            // The inverse modulo is calculated using the Extended Euclidean Algorithm (iterative version)
            // Used to compute integers x and y such that: ax + ny = gcd(a, n).
            // When the gcd is 1, then the inverse of a modulo n exists and it's x.
            // ax + ny = 1
            // ax = 1 + (-y)n
            // ax ≡ 1 (mod n) # x is the inverse of a modulo n

            // If the remainder is 0 the gcd is n right away.
            uint256 remainder = a % n;
            uint256 gcd = n;

            // Therefore the initial coefficients are:
            // ax + ny = gcd(a, n) = n
            // 0a + 1n = n
            int256 x = 0;
            int256 y = 1;

            while (remainder != 0) {
                uint256 quotient = gcd / remainder;

                (gcd, remainder) = (
                    // The old remainder is the next gcd to try.
                    remainder,
                    // Compute the next remainder.
                    // Can't overflow given that (a % gcd) * (gcd // (a % gcd)) <= gcd
                    // where gcd is at most n (capped to type(uint256).max)
                    gcd - remainder * quotient
                );

                (x, y) = (
                    // Increment the coefficient of a.
                    y,
                    // Decrement the coefficient of n.
                    // Can overflow, but the result is casted to uint256 so that the
                    // next value of y is "wrapped around" to a value between 0 and n - 1.
                    x - y * int256(quotient)
                );
            }

            if (gcd != 1) return 0; // No inverse exists.
            return ternary(x < 0, n - uint256(-x), uint256(x)); // Wrap the result if it's negative.
        }
    }

    /**
     * @dev Variant of {invMod}. More efficient, but only works if `p` is known to be a prime greater than `2`.
     *
     * From https://en.wikipedia.org/wiki/Fermat%27s_little_theorem[Fermat's little theorem], we know that if p is
     * prime, then `a**(p-1) ≡ 1 mod p`. As a consequence, we have `a * a**(p-2) ≡ 1 mod p`, which means that
     * `a**(p-2)` is the modular multiplicative inverse of a in Fp.
     *
     * NOTE: this function does NOT check that `p` is a prime greater than `2`.
     */
    function invModPrime(uint256 a, uint256 p) internal view returns (uint256) {
        unchecked {
            return Math.modExp(a, p - 2, p);
        }
    }

    /**
     * @dev Returns the modular exponentiation of the specified base, exponent and modulus (b ** e % m)
     *
     * Requirements:
     * - modulus can't be zero
     * - underlying staticcall to precompile must succeed
     *
     * IMPORTANT: The result is only valid if the underlying call succeeds. When using this function, make
     * sure the chain you're using it on supports the precompiled contract for modular exponentiation
     * at address 0x05 as specified in https://eips.ethereum.org/EIPS/eip-198[EIP-198]. Otherwise,
     * the underlying function will succeed given the lack of a revert, but the result may be incorrectly
     * interpreted as 0.
     */
    function modExp(uint256 b, uint256 e, uint256 m) internal view returns (uint256) {
        (bool success, uint256 result) = tryModExp(b, e, m);
        if (!success) {
            Panic.panic(Panic.DIVISION_BY_ZERO);
        }
        return result;
    }

    /**
     * @dev Returns the modular exponentiation of the specified base, exponent and modulus (b ** e % m).
     * It includes a success flag indicating if the operation succeeded. Operation will be marked as failed if trying
     * to operate modulo 0 or if the underlying precompile reverted.
     *
     * IMPORTANT: The result is only valid if the success flag is true. When using this function, make sure the chain
     * you're using it on supports the precompiled contract for modular exponentiation at address 0x05 as specified in
     * https://eips.ethereum.org/EIPS/eip-198[EIP-198]. Otherwise, the underlying function will succeed given the lack
     * of a revert, but the result may be incorrectly interpreted as 0.
     */
    function tryModExp(uint256 b, uint256 e, uint256 m) internal view returns (bool success, uint256 result) {
        if (m == 0) return (false, 0);
        assembly ("memory-safe") {
            let ptr := mload(0x40)
            // | Offset    | Content    | Content (Hex)                                                      |
            // |-----------|------------|--------------------------------------------------------------------|
            // | 0x00:0x1f | size of b  | 0x0000000000000000000000000000000000000000000000000000000000000020 |
            // | 0x20:0x3f | size of e  | 0x0000000000000000000000000000000000000000000000000000000000000020 |
            // | 0x40:0x5f | size of m  | 0x0000000000000000000000000000000000000000000000000000000000000020 |
            // | 0x60:0x7f | value of b | 0x<.............................................................b> |
            // | 0x80:0x9f | value of e | 0x<.............................................................e> |
            // | 0xa0:0xbf | value of m | 0x<.............................................................m> |
            mstore(ptr, 0x20)
            mstore(add(ptr, 0x20), 0x20)
            mstore(add(ptr, 0x40), 0x20)
            mstore(add(ptr, 0x60), b)
            mstore(add(ptr, 0x80), e)
            mstore(add(ptr, 0xa0), m)

            // Given the result < m, it's guaranteed to fit in 32 bytes,
            // so we can use the memory scratch space located at offset 0.
            success := staticcall(gas(), 0x05, ptr, 0xc0, 0x00, 0x20)
            result := mload(0x00)
        }
    }

    /**
     * @dev Variant of {modExp} that supports inputs of arbitrary length.
     */
    function modExp(bytes memory b, bytes memory e, bytes memory m) internal view returns (bytes memory) {
        (bool success, bytes memory result) = tryModExp(b, e, m);
        if (!success) {
            Panic.panic(Panic.DIVISION_BY_ZERO);
        }
        return result;
    }

    /**
     * @dev Variant of {tryModExp} that supports inputs of arbitrary length.
     */
    function tryModExp(
        bytes memory b,
        bytes memory e,
        bytes memory m
    ) internal view returns (bool success, bytes memory result) {
        if (_zeroBytes(m)) return (false, new bytes(0));

        uint256 mLen = m.length;

        // Encode call args in result and move the free memory pointer
        result = abi.encodePacked(b.length, e.length, mLen, b, e, m);

        assembly ("memory-safe") {
            let dataPtr := add(result, 0x20)
            // Write result on top of args to avoid allocating extra memory.
            success := staticcall(gas(), 0x05, dataPtr, mload(result), dataPtr, mLen)
            // Overwrite the length.
            // result.length > returndatasize() is guaranteed because returndatasize() == m.length
            mstore(result, mLen)
            // Set the memory pointer after the returned data.
            mstore(0x40, add(dataPtr, mLen))
        }
    }

    /**
     * @dev Returns whether the provided byte array is zero.
     */
    function _zeroBytes(bytes memory byteArray) private pure returns (bool) {
        for (uint256 i = 0; i < byteArray.length; ++i) {
            if (byteArray[i] != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @dev Returns the square root of a number. If the number is not a perfect square, the value is rounded
     * towards zero.
     *
     * This method is based on Newton's method for computing square roots; the algorithm is restricted to only
     * using integer operations.
     */
    function sqrt(uint256 a) internal pure returns (uint256) {
        unchecked {
            // Take care of easy edge cases when a == 0 or a == 1
            if (a <= 1) {
                return a;
            }

            // In this function, we use Newton's method to get a root of `f(x) := x² - a`. It involves building a
            // sequence x_n that converges toward sqrt(a). For each iteration x_n, we also define the error between
            // the current value as `ε_n = | x_n - sqrt(a) |`.
            //
            // For our first estimation, we consider `e` the smallest power of 2 which is bigger than the square root
            // of the target. (i.e. `2**(e-1) ≤ sqrt(a) < 2**e`). We know that `e ≤ 128` because `(2¹²⁸)² = 2²⁵⁶` is
            // bigger than any uint256.
            //
            // By noticing that
            // `2**(e-1) ≤ sqrt(a) < 2**e → (2**(e-1))² ≤ a < (2**e)² → 2**(2*e-2) ≤ a < 2**(2*e)`
            // we can deduce that `e - 1` is `log2(a) / 2`. We can thus compute `x_n = 2**(e-1)` using a method similar
            // to the msb function.
            uint256 aa = a;
            uint256 xn = 1;

            if (aa >= (1 << 128)) {
                aa >>= 128;
                xn <<= 64;
            }
            if (aa >= (1 << 64)) {
                aa >>= 64;
                xn <<= 32;
            }
            if (aa >= (1 << 32)) {
                aa >>= 32;
                xn <<= 16;
            }
            if (aa >= (1 << 16)) {
                aa >>= 16;
                xn <<= 8;
            }
            if (aa >= (1 << 8)) {
                aa >>= 8;
                xn <<= 4;
            }
            if (aa >= (1 << 4)) {
                aa >>= 4;
                xn <<= 2;
            }
            if (aa >= (1 << 2)) {
                xn <<= 1;
            }

            // We now have x_n such that `x_n = 2**(e-1) ≤ sqrt(a) < 2**e = 2 * x_n`. This implies ε_n ≤ 2**(e-1).
            //
            // We can refine our estimation by noticing that the middle of that interval minimizes the error.
            // If we move x_n to equal 2**(e-1) + 2**(e-2), then we reduce the error to ε_n ≤ 2**(e-2).
            // This is going to be our x_0 (and ε_0)
            xn = (3 * xn) >> 1; // ε_0 := | x_0 - sqrt(a) | ≤ 2**(e-2)

            // From here, Newton's method give us:
            // x_{n+1} = (x_n + a / x_n) / 2
            //
            // One should note that:
            // x_{n+1}² - a = ((x_n + a / x_n) / 2)² - a
            //              = ((x_n² + a) / (2 * x_n))² - a
            //              = (x_n⁴ + 2 * a * x_n² + a²) / (4 * x_n²) - a
            //              = (x_n⁴ + 2 * a * x_n² + a² - 4 * a * x_n²) / (4 * x_n²)
            //              = (x_n⁴ - 2 * a * x_n² + a²) / (4 * x_n²)
            //              = (x_n² - a)² / (2 * x_n)²
            //              = ((x_n² - a) / (2 * x_n))²
            //              ≥ 0
            // Which proves that for all n ≥ 1, sqrt(a) ≤ x_n
            //
            // This gives us the proof of quadratic convergence of the sequence:
            // ε_{n+1} = | x_{n+1} - sqrt(a) |
            //         = | (x_n + a / x_n) / 2 - sqrt(a) |
            //         = | (x_n² + a - 2*x_n*sqrt(a)) / (2 * x_n) |
            //         = | (x_n - sqrt(a))² / (2 * x_n) |
            //         = | ε_n² / (2 * x_n) |
            //         = ε_n² / | (2 * x_n) |
            //
            // For the first iteration, we have a special case where x_0 is known:
            // ε_1 = ε_0² / | (2 * x_0) |
            //     ≤ (2**(e-2))² / (2 * (2**(e-1) + 2**(e-2)))
            //     ≤ 2**(2*e-4) / (3 * 2**(e-1))
            //     ≤ 2**(e-3) / 3
            //     ≤ 2**(e-3-log2(3))
            //     ≤ 2**(e-4.5)
            //
            // For the following iterations, we use the fact that, 2**(e-1) ≤ sqrt(a) ≤ x_n:
            // ε_{n+1} = ε_n² / | (2 * x_n) |
            //         ≤ (2**(e-k))² / (2 * 2**(e-1))
            //         ≤ 2**(2*e-2*k) / 2**e
            //         ≤ 2**(e-2*k)
            xn = (xn + a / xn) >> 1; // ε_1 := | x_1 - sqrt(a) | ≤ 2**(e-4.5)  -- special case, see above
            xn = (xn + a / xn) >> 1; // ε_2 := | x_2 - sqrt(a) | ≤ 2**(e-9)    -- general case with k = 4.5
            xn = (xn + a / xn) >> 1; // ε_3 := | x_3 - sqrt(a) | ≤ 2**(e-18)   -- general case with k = 9
            xn = (xn + a / xn) >> 1; // ε_4 := | x_4 - sqrt(a) | ≤ 2**(e-36)   -- general case with k = 18
            xn = (xn + a / xn) >> 1; // ε_5 := | x_5 - sqrt(a) | ≤ 2**(e-72)   -- general case with k = 36
            xn = (xn + a / xn) >> 1; // ε_6 := | x_6 - sqrt(a) | ≤ 2**(e-144)  -- general case with k = 72

            // Because e ≤ 128 (as discussed during the first estimation phase), we know have reached a precision
            // ε_6 ≤ 2**(e-144) < 1. Given we're operating on integers, then we can ensure that xn is now either
            // sqrt(a) or sqrt(a) + 1.
            return xn - SafeCast.toUint(xn > a / xn);
        }
    }

    /**
     * @dev Calculates sqrt(a), following the selected rounding direction.
     */
    function sqrt(uint256 a, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = sqrt(a);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && result * result < a);
        }
    }

    /**
     * @dev Return the log in base 2 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log2(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        uint256 exp;
        unchecked {
            exp = 128 * SafeCast.toUint(value > (1 << 128) - 1);
            value >>= exp;
            result += exp;

            exp = 64 * SafeCast.toUint(value > (1 << 64) - 1);
            value >>= exp;
            result += exp;

            exp = 32 * SafeCast.toUint(value > (1 << 32) - 1);
            value >>= exp;
            result += exp;

            exp = 16 * SafeCast.toUint(value > (1 << 16) - 1);
            value >>= exp;
            result += exp;

            exp = 8 * SafeCast.toUint(value > (1 << 8) - 1);
            value >>= exp;
            result += exp;

            exp = 4 * SafeCast.toUint(value > (1 << 4) - 1);
            value >>= exp;
            result += exp;

            exp = 2 * SafeCast.toUint(value > (1 << 2) - 1);
            value >>= exp;
            result += exp;

            result += SafeCast.toUint(value > 1);
        }
        return result;
    }

    /**
     * @dev Return the log in base 2, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log2(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log2(value);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && 1 << result < value);
        }
    }

    /**
     * @dev Return the log in base 10 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     */
    function log10(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >= 10 ** 64) {
                value /= 10 ** 64;
                result += 64;
            }
            if (value >= 10 ** 32) {
                value /= 10 ** 32;
                result += 32;
            }
            if (value >= 10 ** 16) {
                value /= 10 ** 16;
                result += 16;
            }
            if (value >= 10 ** 8) {
                value /= 10 ** 8;
                result += 8;
            }
            if (value >= 10 ** 4) {
                value /= 10 ** 4;
                result += 4;
            }
            if (value >= 10 ** 2) {
                value /= 10 ** 2;
                result += 2;
            }
            if (value >= 10 ** 1) {
                result += 1;
            }
        }
        return result;
    }

    /**
     * @dev Return the log in base 10, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log10(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log10(value);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && 10 ** result < value);
        }
    }

    /**
     * @dev Return the log in base 256 of a positive value rounded towards zero.
     * Returns 0 if given 0.
     *
     * Adding one to the result gives the number of pairs of hex symbols needed to represent `value` as a hex string.
     */
    function log256(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        uint256 isGt;
        unchecked {
            isGt = SafeCast.toUint(value > (1 << 128) - 1);
            value >>= isGt * 128;
            result += isGt * 16;

            isGt = SafeCast.toUint(value > (1 << 64) - 1);
            value >>= isGt * 64;
            result += isGt * 8;

            isGt = SafeCast.toUint(value > (1 << 32) - 1);
            value >>= isGt * 32;
            result += isGt * 4;

            isGt = SafeCast.toUint(value > (1 << 16) - 1);
            value >>= isGt * 16;
            result += isGt * 2;

            result += SafeCast.toUint(value > (1 << 8) - 1);
        }
        return result;
    }

    /**
     * @dev Return the log in base 256, following the selected rounding direction, of a positive value.
     * Returns 0 if given 0.
     */
    function log256(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log256(value);
            return result + SafeCast.toUint(unsignedRoundsUp(rounding) && 1 << (result << 3) < value);
        }
    }

    /**
     * @dev Returns whether a provided rounding mode is considered rounding up for unsigned integers.
     */
    function unsignedRoundsUp(Rounding rounding) internal pure returns (bool) {
        return uint8(rounding) % 2 == 1;
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/math/SignedMath.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/math/SignedMath.sol)

pragma solidity ^0.8.20;

import {SafeCast} from "./SafeCast.sol";

/**
 * @dev Standard signed math utilities missing in the Solidity language.
 */
library SignedMath {
    /**
     * @dev Branchless ternary evaluation for `a ? b : c`. Gas costs are constant.
     *
     * IMPORTANT: This function may reduce bytecode size and consume less gas when used standalone.
     * However, the compiler may optimize Solidity ternary operations (i.e. `a ? b : c`) to only compute
     * one branch when needed, making this function more expensive.
     */
    function ternary(bool condition, int256 a, int256 b) internal pure returns (int256) {
        unchecked {
            // branchless ternary works because:
            // b ^ (a ^ b) == a
            // b ^ 0 == b
            return b ^ ((a ^ b) * int256(SafeCast.toUint(condition)));
        }
    }

    /**
     * @dev Returns the largest of two signed numbers.
     */
    function max(int256 a, int256 b) internal pure returns (int256) {
        return ternary(a > b, a, b);
    }

    /**
     * @dev Returns the smallest of two signed numbers.
     */
    function min(int256 a, int256 b) internal pure returns (int256) {
        return ternary(a < b, a, b);
    }

    /**
     * @dev Returns the average of two signed numbers without overflow.
     * The result is rounded towards zero.
     */
    function average(int256 a, int256 b) internal pure returns (int256) {
        // Formula from the book "Hacker's Delight"
        int256 x = (a & b) + ((a ^ b) >> 1);
        return x + (int256(uint256(x) >> 255) & (a ^ b));
    }

    /**
     * @dev Returns the absolute unsigned value of a signed value.
     */
    function abs(int256 n) internal pure returns (uint256) {
        unchecked {
            // Formula from the "Bit Twiddling Hacks" by Sean Eron Anderson.
            // Since `n` is a signed integer, the generated bytecode will use the SAR opcode to perform the right shift,
            // taking advantage of the most significant (or "sign" bit) in two's complement representation.
            // This opcode adds new most significant bits set to the value of the previous most significant bit. As a result,
            // the mask will either be `bytes32(0)` (if n is positive) or `~bytes32(0)` (if n is negative).
            int256 mask = n >> 255;

            // A `bytes32(0)` mask leaves the input unchanged, while a `~bytes32(0)` mask complements it.
            return uint256((n + mask) ^ mask);
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/token/ERC721/IERC721Receiver.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (token/ERC721/IERC721Receiver.sol)

pragma solidity ^0.8.20;

/**
 * @title ERC-721 token receiver interface
 * @dev Interface for any contract that wants to support safeTransfers
 * from ERC-721 asset contracts.
 */
interface IERC721Receiver {
    /**
     * @dev Whenever an {IERC721} `tokenId` token is transferred to this contract via {IERC721-safeTransferFrom}
     * by `operator` from `from`, this function is called.
     *
     * It must return its Solidity selector to confirm the token transfer.
     * If any other value is returned or the interface is not implemented by the recipient, the transfer will be
     * reverted.
     *
     * The selector can be obtained in Solidity with `IERC721Receiver.onERC721Received.selector`.
     */
    function onERC721Received(
        address operator,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external returns (bytes4);
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/Panic.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/Panic.sol)

pragma solidity ^0.8.20;

/**
 * @dev Helper library for emitting standardized panic codes.
 *
 * ```solidity
 * contract Example {
 *      using Panic for uint256;
 *
 *      // Use any of the declared internal constants
 *      function foo() { Panic.GENERIC.panic(); }
 *
 *      // Alternatively
 *      function foo() { Panic.panic(Panic.GENERIC); }
 * }
 * ```
 *
 * Follows the list from https://github.com/ethereum/solidity/blob/v0.8.24/libsolutil/ErrorCodes.h[libsolutil].
 *
 * _Available since v5.1._
 */
// slither-disable-next-line unused-state
library Panic {
    /// @dev generic / unspecified error
    uint256 internal constant GENERIC = 0x00;
    /// @dev used by the assert() builtin
    uint256 internal constant ASSERT = 0x01;
    /// @dev arithmetic underflow or overflow
    uint256 internal constant UNDER_OVERFLOW = 0x11;
    /// @dev division or modulo by zero
    uint256 internal constant DIVISION_BY_ZERO = 0x12;
    /// @dev enum conversion error
    uint256 internal constant ENUM_CONVERSION_ERROR = 0x21;
    /// @dev invalid encoding in storage
    uint256 internal constant STORAGE_ENCODING_ERROR = 0x22;
    /// @dev empty array pop
    uint256 internal constant EMPTY_ARRAY_POP = 0x31;
    /// @dev array out of bounds access
    uint256 internal constant ARRAY_OUT_OF_BOUNDS = 0x32;
    /// @dev resource error (too large allocation or too large array)
    uint256 internal constant RESOURCE_ERROR = 0x41;
    /// @dev calling invalid internal function
    uint256 internal constant INVALID_INTERNAL_FUNCTION = 0x51;

    /// @dev Reverts with a panic code. Recommended to use with
    /// the internal constants with predefined codes.
    function panic(uint256 code) internal pure {
        assembly ("memory-safe") {
            mstore(0x00, 0x4e487b71)
            mstore(0x20, code)
            revert(0x1c, 0x24)
        }
    }
}


// ===== FILE: lib/openzeppelin-contracts/contracts/utils/math/SafeCast.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.1.0) (utils/math/SafeCast.sol)
// This file was procedurally generated from scripts/generate/templates/SafeCast.js.

pragma solidity ^0.8.20;

/**
 * @dev Wrappers over Solidity's uintXX/intXX/bool casting operators with added overflow
 * checks.
 *
 * Downcasting from uint256/int256 in Solidity does not revert on overflow. This can
 * easily result in undesired exploitation or bugs, since developers usually
 * assume that overflows raise errors. `SafeCast` restores this intuition by
 * reverting the transaction when such an operation overflows.
 *
 * Using this library instead of the unchecked operations eliminates an entire
 * class of bugs, so it's recommended to use it always.
 */
library SafeCast {
    /**
     * @dev Value doesn't fit in an uint of `bits` size.
     */
    error SafeCastOverflowedUintDowncast(uint8 bits, uint256 value);

    /**
     * @dev An int value doesn't fit in an uint of `bits` size.
     */
    error SafeCastOverflowedIntToUint(int256 value);

    /**
     * @dev Value doesn't fit in an int of `bits` size.
     */
    error SafeCastOverflowedIntDowncast(uint8 bits, int256 value);

    /**
     * @dev An uint value doesn't fit in an int of `bits` size.
     */
    error SafeCastOverflowedUintToInt(uint256 value);

    /**
     * @dev Returns the downcasted uint248 from uint256, reverting on
     * overflow (when the input is greater than largest uint248).
     *
     * Counterpart to Solidity's `uint248` operator.
     *
     * Requirements:
     *
     * - input must fit into 248 bits
     */
    function toUint248(uint256 value) internal pure returns (uint248) {
        if (value > type(uint248).max) {
            revert SafeCastOverflowedUintDowncast(248, value);
        }
        return uint248(value);
    }

    /**
     * @dev Returns the downcasted uint240 from uint256, reverting on
     * overflow (when the input is greater than largest uint240).
     *
     * Counterpart to Solidity's `uint240` operator.
     *
     * Requirements:
     *
     * - input must fit into 240 bits
     */
    function toUint240(uint256 value) internal pure returns (uint240) {
        if (value > type(uint240).max) {
            revert SafeCastOverflowedUintDowncast(240, value);
        }
        return uint240(value);
    }

    /**
     * @dev Returns the downcasted uint232 from uint256, reverting on
     * overflow (when the input is greater than largest uint232).
     *
     * Counterpart to Solidity's `uint232` operator.
     *
     * Requirements:
     *
     * - input must fit into 232 bits
     */
    function toUint232(uint256 value) internal pure returns (uint232) {
        if (value > type(uint232).max) {
            revert SafeCastOverflowedUintDowncast(232, value);
        }
        return uint232(value);
    }

    /**
     * @dev Returns the downcasted uint224 from uint256, reverting on
     * overflow (when the input is greater than largest uint224).
     *
     * Counterpart to Solidity's `uint224` operator.
     *
     * Requirements:
     *
     * - input must fit into 224 bits
     */
    function toUint224(uint256 value) internal pure returns (uint224) {
        if (value > type(uint224).max) {
            revert SafeCastOverflowedUintDowncast(224, value);
        }
        return uint224(value);
    }

    /**
     * @dev Returns the downcasted uint216 from uint256, reverting on
     * overflow (when the input is greater than largest uint216).
     *
     * Counterpart to Solidity's `uint216` operator.
     *
     * Requirements:
     *
     * - input must fit into 216 bits
     */
    function toUint216(uint256 value) internal pure returns (uint216) {
        if (value > type(uint216).max) {
            revert SafeCastOverflowedUintDowncast(216, value);
        }
        return uint216(value);
    }

    /**
     * @dev Returns the downcasted uint208 from uint256, reverting on
     * overflow (when the input is greater than largest uint208).
     *
     * Counterpart to Solidity's `uint208` operator.
     *
     * Requirements:
     *
     * - input must fit into 208 bits
     */
    function toUint208(uint256 value) internal pure returns (uint208) {
        if (value > type(uint208).max) {
            revert SafeCastOverflowedUintDowncast(208, value);
        }
        return uint208(value);
    }

    /**
     * @dev Returns the downcasted uint200 from uint256, reverting on
     * overflow (when the input is greater than largest uint200).
     *
     * Counterpart to Solidity's `uint200` operator.
     *
     * Requirements:
     *
     * - input must fit into 200 bits
     */
    function toUint200(uint256 value) internal pure returns (uint200) {
        if (value > type(uint200).max) {
            revert SafeCastOverflowedUintDowncast(200, value);
        }
        return uint200(value);
    }

    /**
     * @dev Returns the downcasted uint192 from uint256, reverting on
     * overflow (when the input is greater than largest uint192).
     *
     * Counterpart to Solidity's `uint192` operator.
     *
     * Requirements:
     *
     * - input must fit into 192 bits
     */
    function toUint192(uint256 value) internal pure returns (uint192) {
        if (value > type(uint192).max) {
            revert SafeCastOverflowedUintDowncast(192, value);
        }
        return uint192(value);
    }

    /**
     * @dev Returns the downcasted uint184 from uint256, reverting on
     * overflow (when the input is greater than largest uint184).
     *
     * Counterpart to Solidity's `uint184` operator.
     *
     * Requirements:
     *
     * - input must fit into 184 bits
     */
    function toUint184(uint256 value) internal pure returns (uint184) {
        if (value > type(uint184).max) {
            revert SafeCastOverflowedUintDowncast(184, value);
        }
        return uint184(value);
    }

    /**
     * @dev Returns the downcasted uint176 from uint256, reverting on
     * overflow (when the input is greater than largest uint176).
     *
     * Counterpart to Solidity's `uint176` operator.
     *
     * Requirements:
     *
     * - input must fit into 176 bits
     */
    function toUint176(uint256 value) internal pure returns (uint176) {
        if (value > type(uint176).max) {
            revert SafeCastOverflowedUintDowncast(176, value);
        }
        return uint176(value);
    }

    /**
     * @dev Returns the downcasted uint168 from uint256, reverting on
     * overflow (when the input is greater than largest uint168).
     *
     * Counterpart to Solidity's `uint168` operator.
     *
     * Requirements:
     *
     * - input must fit into 168 bits
     */
    function toUint168(uint256 value) internal pure returns (uint168) {
        if (value > type(uint168).max) {
            revert SafeCastOverflowedUintDowncast(168, value);
        }
        return uint168(value);
    }

    /**
     * @dev Returns the downcasted uint160 from uint256, reverting on
     * overflow (when the input is greater than largest uint160).
     *
     * Counterpart to Solidity's `uint160` operator.
     *
     * Requirements:
     *
     * - input must fit into 160 bits
     */
    function toUint160(uint256 value) internal pure returns (uint160) {
        if (value > type(uint160).max) {
            revert SafeCastOverflowedUintDowncast(160, value);
        }
        return uint160(value);
    }

    /**
     * @dev Returns the downcasted uint152 from uint256, reverting on
     * overflow (when the input is greater than largest uint152).
     *
     * Counterpart to Solidity's `uint152` operator.
     *
     * Requirements:
     *
     * - input must fit into 152 bits
     */
    function toUint152(uint256 value) internal pure returns (uint152) {
        if (value > type(uint152).max) {
            revert SafeCastOverflowedUintDowncast(152, value);
        }
        return uint152(value);
    }

    /**
     * @dev Returns the downcasted uint144 from uint256, reverting on
     * overflow (when the input is greater than largest uint144).
     *
     * Counterpart to Solidity's `uint144` operator.
     *
     * Requirements:
     *
     * - input must fit into 144 bits
     */
    function toUint144(uint256 value) internal pure returns (uint144) {
        if (value > type(uint144).max) {
            revert SafeCastOverflowedUintDowncast(144, value);
        }
        return uint144(value);
    }

    /**
     * @dev Returns the downcasted uint136 from uint256, reverting on
     * overflow (when the input is greater than largest uint136).
     *
     * Counterpart to Solidity's `uint136` operator.
     *
     * Requirements:
     *
     * - input must fit into 136 bits
     */
    function toUint136(uint256 value) internal pure returns (uint136) {
        if (value > type(uint136).max) {
            revert SafeCastOverflowedUintDowncast(136, value);
        }
        return uint136(value);
    }

    /**
     * @dev Returns the downcasted uint128 from uint256, reverting on
     * overflow (when the input is greater than largest uint128).
     *
     * Counterpart to Solidity's `uint128` operator.
     *
     * Requirements:
     *
     * - input must fit into 128 bits
     */
    function toUint128(uint256 value) internal pure returns (uint128) {
        if (value > type(uint128).max) {
            revert SafeCastOverflowedUintDowncast(128, value);
        }
        return uint128(value);
    }

    /**
     * @dev Returns the downcasted uint120 from uint256, reverting on
     * overflow (when the input is greater than largest uint120).
     *
     * Counterpart to Solidity's `uint120` operator.
     *
     * Requirements:
     *
     * - input must fit into 120 bits
     */
    function toUint120(uint256 value) internal pure returns (uint120) {
        if (value > type(uint120).max) {
            revert SafeCastOverflowedUintDowncast(120, value);
        }
        return uint120(value);
    }

    /**
     * @dev Returns the downcasted uint112 from uint256, reverting on
     * overflow (when the input is greater than largest uint112).
     *
     * Counterpart to Solidity's `uint112` operator.
     *
     * Requirements:
     *
     * - input must fit into 112 bits
     */
    function toUint112(uint256 value) internal pure returns (uint112) {
        if (value > type(uint112).max) {
            revert SafeCastOverflowedUintDowncast(112, value);
        }
        return uint112(value);
    }

    /**
     * @dev Returns the downcasted uint104 from uint256, reverting on
     * overflow (when the input is greater than largest uint104).
     *
     * Counterpart to Solidity's `uint104` operator.
     *
     * Requirements:
     *
     * - input must fit into 104 bits
     */
    function toUint104(uint256 value) internal pure returns (uint104) {
        if (value > type(uint104).max) {
            revert SafeCastOverflowedUintDowncast(104, value);
        }
        return uint104(value);
    }

    /**
     * @dev Returns the downcasted uint96 from uint256, reverting on
     * overflow (when the input is greater than largest uint96).
     *
     * Counterpart to Solidity's `uint96` operator.
     *
     * Requirements:
     *
     * - input must fit into 96 bits
     */
    function toUint96(uint256 value) internal pure returns (uint96) {
        if (value > type(uint96).max) {
            revert SafeCastOverflowedUintDowncast(96, value);
        }
        return uint96(value);
    }

    /**
     * @dev Returns the downcasted uint88 from uint256, reverting on
     * overflow (when the input is greater than largest uint88).
     *
     * Counterpart to Solidity's `uint88` operator.
     *
     * Requirements:
     *
     * - input must fit into 88 bits
     */
    function toUint88(uint256 value) internal pure returns (uint88) {
        if (value > type(uint88).max) {
            revert SafeCastOverflowedUintDowncast(88, value);
        }
        return uint88(value);
    }

    /**
     * @dev Returns the downcasted uint80 from uint256, reverting on
     * overflow (when the input is greater than largest uint80).
     *
     * Counterpart to Solidity's `uint80` operator.
     *
     * Requirements:
     *
     * - input must fit into 80 bits
     */
    function toUint80(uint256 value) internal pure returns (uint80) {
        if (value > type(uint80).max) {
            revert SafeCastOverflowedUintDowncast(80, value);
        }
        return uint80(value);
    }

    /**
     * @dev Returns the downcasted uint72 from uint256, reverting on
     * overflow (when the input is greater than largest uint72).
     *
     * Counterpart to Solidity's `uint72` operator.
     *
     * Requirements:
     *
     * - input must fit into 72 bits
     */
    function toUint72(uint256 value) internal pure returns (uint72) {
        if (value > type(uint72).max) {
            revert SafeCastOverflowedUintDowncast(72, value);
        }
        return uint72(value);
    }

    /**
     * @dev Returns the downcasted uint64 from uint256, reverting on
     * overflow (when the input is greater than largest uint64).
     *
     * Counterpart to Solidity's `uint64` operator.
     *
     * Requirements:
     *
     * - input must fit into 64 bits
     */
    function toUint64(uint256 value) internal pure returns (uint64) {
        if (value > type(uint64).max) {
            revert SafeCastOverflowedUintDowncast(64, value);
        }
        return uint64(value);
    }

    /**
     * @dev Returns the downcasted uint56 from uint256, reverting on
     * overflow (when the input is greater than largest uint56).
     *
     * Counterpart to Solidity's `uint56` operator.
     *
     * Requirements:
     *
     * - input must fit into 56 bits
     */
    function toUint56(uint256 value) internal pure returns (uint56) {
        if (value > type(uint56).max) {
            revert SafeCastOverflowedUintDowncast(56, value);
        }
        return uint56(value);
    }

    /**
     * @dev Returns the downcasted uint48 from uint256, reverting on
     * overflow (when the input is greater than largest uint48).
     *
     * Counterpart to Solidity's `uint48` operator.
     *
     * Requirements:
     *
     * - input must fit into 48 bits
     */
    function toUint48(uint256 value) internal pure returns (uint48) {
        if (value > type(uint48).max) {
            revert SafeCastOverflowedUintDowncast(48, value);
        }
        return uint48(value);
    }

    /**
     * @dev Returns the downcasted uint40 from uint256, reverting on
     * overflow (when the input is greater than largest uint40).
     *
     * Counterpart to Solidity's `uint40` operator.
     *
     * Requirements:
     *
     * - input must fit into 40 bits
     */
    function toUint40(uint256 value) internal pure returns (uint40) {
        if (value > type(uint40).max) {
            revert SafeCastOverflowedUintDowncast(40, value);
        }
        return uint40(value);
    }

    /**
     * @dev Returns the downcasted uint32 from uint256, reverting on
     * overflow (when the input is greater than largest uint32).
     *
     * Counterpart to Solidity's `uint32` operator.
     *
     * Requirements:
     *
     * - input must fit into 32 bits
     */
    function toUint32(uint256 value) internal pure returns (uint32) {
        if (value > type(uint32).max) {
            revert SafeCastOverflowedUintDowncast(32, value);
        }
        return uint32(value);
    }

    /**
     * @dev Returns the downcasted uint24 from uint256, reverting on
     * overflow (when the input is greater than largest uint24).
     *
     * Counterpart to Solidity's `uint24` operator.
     *
     * Requirements:
     *
     * - input must fit into 24 bits
     */
    function toUint24(uint256 value) internal pure returns (uint24) {
        if (value > type(uint24).max) {
            revert SafeCastOverflowedUintDowncast(24, value);
        }
        return uint24(value);
    }

    /**
     * @dev Returns the downcasted uint16 from uint256, reverting on
     * overflow (when the input is greater than largest uint16).
     *
     * Counterpart to Solidity's `uint16` operator.
     *
     * Requirements:
     *
     * - input must fit into 16 bits
     */
    function toUint16(uint256 value) internal pure returns (uint16) {
        if (value > type(uint16).max) {
            revert SafeCastOverflowedUintDowncast(16, value);
        }
        return uint16(value);
    }

    /**
     * @dev Returns the downcasted uint8 from uint256, reverting on
     * overflow (when the input is greater than largest uint8).
     *
     * Counterpart to Solidity's `uint8` operator.
     *
     * Requirements:
     *
     * - input must fit into 8 bits
     */
    function toUint8(uint256 value) internal pure returns (uint8) {
        if (value > type(uint8).max) {
            revert SafeCastOverflowedUintDowncast(8, value);
        }
        return uint8(value);
    }

    /**
     * @dev Converts a signed int256 into an unsigned uint256.
     *
     * Requirements:
     *
     * - input must be greater than or equal to 0.
     */
    function toUint256(int256 value) internal pure returns (uint256) {
        if (value < 0) {
            revert SafeCastOverflowedIntToUint(value);
        }
        return uint256(value);
    }

    /**
     * @dev Returns the downcasted int248 from int256, reverting on
     * overflow (when the input is less than smallest int248 or
     * greater than largest int248).
     *
     * Counterpart to Solidity's `int248` operator.
     *
     * Requirements:
     *
     * - input must fit into 248 bits
     */
    function toInt248(int256 value) internal pure returns (int248 downcasted) {
        downcasted = int248(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(248, value);
        }
    }

    /**
     * @dev Returns the downcasted int240 from int256, reverting on
     * overflow (when the input is less than smallest int240 or
     * greater than largest int240).
     *
     * Counterpart to Solidity's `int240` operator.
     *
     * Requirements:
     *
     * - input must fit into 240 bits
     */
    function toInt240(int256 value) internal pure returns (int240 downcasted) {
        downcasted = int240(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(240, value);
        }
    }

    /**
     * @dev Returns the downcasted int232 from int256, reverting on
     * overflow (when the input is less than smallest int232 or
     * greater than largest int232).
     *
     * Counterpart to Solidity's `int232` operator.
     *
     * Requirements:
     *
     * - input must fit into 232 bits
     */
    function toInt232(int256 value) internal pure returns (int232 downcasted) {
        downcasted = int232(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(232, value);
        }
    }

    /**
     * @dev Returns the downcasted int224 from int256, reverting on
     * overflow (when the input is less than smallest int224 or
     * greater than largest int224).
     *
     * Counterpart to Solidity's `int224` operator.
     *
     * Requirements:
     *
     * - input must fit into 224 bits
     */
    function toInt224(int256 value) internal pure returns (int224 downcasted) {
        downcasted = int224(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(224, value);
        }
    }

    /**
     * @dev Returns the downcasted int216 from int256, reverting on
     * overflow (when the input is less than smallest int216 or
     * greater than largest int216).
     *
     * Counterpart to Solidity's `int216` operator.
     *
     * Requirements:
     *
     * - input must fit into 216 bits
     */
    function toInt216(int256 value) internal pure returns (int216 downcasted) {
        downcasted = int216(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(216, value);
        }
    }

    /**
     * @dev Returns the downcasted int208 from int256, reverting on
     * overflow (when the input is less than smallest int208 or
     * greater than largest int208).
     *
     * Counterpart to Solidity's `int208` operator.
     *
     * Requirements:
     *
     * - input must fit into 208 bits
     */
    function toInt208(int256 value) internal pure returns (int208 downcasted) {
        downcasted = int208(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(208, value);
        }
    }

    /**
     * @dev Returns the downcasted int200 from int256, reverting on
     * overflow (when the input is less than smallest int200 or
     * greater than largest int200).
     *
     * Counterpart to Solidity's `int200` operator.
     *
     * Requirements:
     *
     * - input must fit into 200 bits
     */
    function toInt200(int256 value) internal pure returns (int200 downcasted) {
        downcasted = int200(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(200, value);
        }
    }

    /**
     * @dev Returns the downcasted int192 from int256, reverting on
     * overflow (when the input is less than smallest int192 or
     * greater than largest int192).
     *
     * Counterpart to Solidity's `int192` operator.
     *
     * Requirements:
     *
     * - input must fit into 192 bits
     */
    function toInt192(int256 value) internal pure returns (int192 downcasted) {
        downcasted = int192(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(192, value);
        }
    }

    /**
     * @dev Returns the downcasted int184 from int256, reverting on
     * overflow (when the input is less than smallest int184 or
     * greater than largest int184).
     *
     * Counterpart to Solidity's `int184` operator.
     *
     * Requirements:
     *
     * - input must fit into 184 bits
     */
    function toInt184(int256 value) internal pure returns (int184 downcasted) {
        downcasted = int184(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(184, value);
        }
    }

    /**
     * @dev Returns the downcasted int176 from int256, reverting on
     * overflow (when the input is less than smallest int176 or
     * greater than largest int176).
     *
     * Counterpart to Solidity's `int176` operator.
     *
     * Requirements:
     *
     * - input must fit into 176 bits
     */
    function toInt176(int256 value) internal pure returns (int176 downcasted) {
        downcasted = int176(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(176, value);
        }
    }

    /**
     * @dev Returns the downcasted int168 from int256, reverting on
     * overflow (when the input is less than smallest int168 or
     * greater than largest int168).
     *
     * Counterpart to Solidity's `int168` operator.
     *
     * Requirements:
     *
     * - input must fit into 168 bits
     */
    function toInt168(int256 value) internal pure returns (int168 downcasted) {
        downcasted = int168(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(168, value);
        }
    }

    /**
     * @dev Returns the downcasted int160 from int256, reverting on
     * overflow (when the input is less than smallest int160 or
     * greater than largest int160).
     *
     * Counterpart to Solidity's `int160` operator.
     *
     * Requirements:
     *
     * - input must fit into 160 bits
     */
    function toInt160(int256 value) internal pure returns (int160 downcasted) {
        downcasted = int160(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(160, value);
        }
    }

    /**
     * @dev Returns the downcasted int152 from int256, reverting on
     * overflow (when the input is less than smallest int152 or
     * greater than largest int152).
     *
     * Counterpart to Solidity's `int152` operator.
     *
     * Requirements:
     *
     * - input must fit into 152 bits
     */
    function toInt152(int256 value) internal pure returns (int152 downcasted) {
        downcasted = int152(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(152, value);
        }
    }

    /**
     * @dev Returns the downcasted int144 from int256, reverting on
     * overflow (when the input is less than smallest int144 or
     * greater than largest int144).
     *
     * Counterpart to Solidity's `int144` operator.
     *
     * Requirements:
     *
     * - input must fit into 144 bits
     */
    function toInt144(int256 value) internal pure returns (int144 downcasted) {
        downcasted = int144(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(144, value);
        }
    }

    /**
     * @dev Returns the downcasted int136 from int256, reverting on
     * overflow (when the input is less than smallest int136 or
     * greater than largest int136).
     *
     * Counterpart to Solidity's `int136` operator.
     *
     * Requirements:
     *
     * - input must fit into 136 bits
     */
    function toInt136(int256 value) internal pure returns (int136 downcasted) {
        downcasted = int136(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(136, value);
        }
    }

    /**
     * @dev Returns the downcasted int128 from int256, reverting on
     * overflow (when the input is less than smallest int128 or
     * greater than largest int128).
     *
     * Counterpart to Solidity's `int128` operator.
     *
     * Requirements:
     *
     * - input must fit into 128 bits
     */
    function toInt128(int256 value) internal pure returns (int128 downcasted) {
        downcasted = int128(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(128, value);
        }
    }

    /**
     * @dev Returns the downcasted int120 from int256, reverting on
     * overflow (when the input is less than smallest int120 or
     * greater than largest int120).
     *
     * Counterpart to Solidity's `int120` operator.
     *
     * Requirements:
     *
     * - input must fit into 120 bits
     */
    function toInt120(int256 value) internal pure returns (int120 downcasted) {
        downcasted = int120(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(120, value);
        }
    }

    /**
     * @dev Returns the downcasted int112 from int256, reverting on
     * overflow (when the input is less than smallest int112 or
     * greater than largest int112).
     *
     * Counterpart to Solidity's `int112` operator.
     *
     * Requirements:
     *
     * - input must fit into 112 bits
     */
    function toInt112(int256 value) internal pure returns (int112 downcasted) {
        downcasted = int112(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(112, value);
        }
    }

    /**
     * @dev Returns the downcasted int104 from int256, reverting on
     * overflow (when the input is less than smallest int104 or
     * greater than largest int104).
     *
     * Counterpart to Solidity's `int104` operator.
     *
     * Requirements:
     *
     * - input must fit into 104 bits
     */
    function toInt104(int256 value) internal pure returns (int104 downcasted) {
        downcasted = int104(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(104, value);
        }
    }

    /**
     * @dev Returns the downcasted int96 from int256, reverting on
     * overflow (when the input is less than smallest int96 or
     * greater than largest int96).
     *
     * Counterpart to Solidity's `int96` operator.
     *
     * Requirements:
     *
     * - input must fit into 96 bits
     */
    function toInt96(int256 value) internal pure returns (int96 downcasted) {
        downcasted = int96(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(96, value);
        }
    }

    /**
     * @dev Returns the downcasted int88 from int256, reverting on
     * overflow (when the input is less than smallest int88 or
     * greater than largest int88).
     *
     * Counterpart to Solidity's `int88` operator.
     *
     * Requirements:
     *
     * - input must fit into 88 bits
     */
    function toInt88(int256 value) internal pure returns (int88 downcasted) {
        downcasted = int88(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(88, value);
        }
    }

    /**
     * @dev Returns the downcasted int80 from int256, reverting on
     * overflow (when the input is less than smallest int80 or
     * greater than largest int80).
     *
     * Counterpart to Solidity's `int80` operator.
     *
     * Requirements:
     *
     * - input must fit into 80 bits
     */
    function toInt80(int256 value) internal pure returns (int80 downcasted) {
        downcasted = int80(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(80, value);
        }
    }

    /**
     * @dev Returns the downcasted int72 from int256, reverting on
     * overflow (when the input is less than smallest int72 or
     * greater than largest int72).
     *
     * Counterpart to Solidity's `int72` operator.
     *
     * Requirements:
     *
     * - input must fit into 72 bits
     */
    function toInt72(int256 value) internal pure returns (int72 downcasted) {
        downcasted = int72(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(72, value);
        }
    }

    /**
     * @dev Returns the downcasted int64 from int256, reverting on
     * overflow (when the input is less than smallest int64 or
     * greater than largest int64).
     *
     * Counterpart to Solidity's `int64` operator.
     *
     * Requirements:
     *
     * - input must fit into 64 bits
     */
    function toInt64(int256 value) internal pure returns (int64 downcasted) {
        downcasted = int64(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(64, value);
        }
    }

    /**
     * @dev Returns the downcasted int56 from int256, reverting on
     * overflow (when the input is less than smallest int56 or
     * greater than largest int56).
     *
     * Counterpart to Solidity's `int56` operator.
     *
     * Requirements:
     *
     * - input must fit into 56 bits
     */
    function toInt56(int256 value) internal pure returns (int56 downcasted) {
        downcasted = int56(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(56, value);
        }
    }

    /**
     * @dev Returns the downcasted int48 from int256, reverting on
     * overflow (when the input is less than smallest int48 or
     * greater than largest int48).
     *
     * Counterpart to Solidity's `int48` operator.
     *
     * Requirements:
     *
     * - input must fit into 48 bits
     */
    function toInt48(int256 value) internal pure returns (int48 downcasted) {
        downcasted = int48(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(48, value);
        }
    }

    /**
     * @dev Returns the downcasted int40 from int256, reverting on
     * overflow (when the input is less than smallest int40 or
     * greater than largest int40).
     *
     * Counterpart to Solidity's `int40` operator.
     *
     * Requirements:
     *
     * - input must fit into 40 bits
     */
    function toInt40(int256 value) internal pure returns (int40 downcasted) {
        downcasted = int40(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(40, value);
        }
    }

    /**
     * @dev Returns the downcasted int32 from int256, reverting on
     * overflow (when the input is less than smallest int32 or
     * greater than largest int32).
     *
     * Counterpart to Solidity's `int32` operator.
     *
     * Requirements:
     *
     * - input must fit into 32 bits
     */
    function toInt32(int256 value) internal pure returns (int32 downcasted) {
        downcasted = int32(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(32, value);
        }
    }

    /**
     * @dev Returns the downcasted int24 from int256, reverting on
     * overflow (when the input is less than smallest int24 or
     * greater than largest int24).
     *
     * Counterpart to Solidity's `int24` operator.
     *
     * Requirements:
     *
     * - input must fit into 24 bits
     */
    function toInt24(int256 value) internal pure returns (int24 downcasted) {
        downcasted = int24(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(24, value);
        }
    }

    /**
     * @dev Returns the downcasted int16 from int256, reverting on
     * overflow (when the input is less than smallest int16 or
     * greater than largest int16).
     *
     * Counterpart to Solidity's `int16` operator.
     *
     * Requirements:
     *
     * - input must fit into 16 bits
     */
    function toInt16(int256 value) internal pure returns (int16 downcasted) {
        downcasted = int16(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(16, value);
        }
    }

    /**
     * @dev Returns the downcasted int8 from int256, reverting on
     * overflow (when the input is less than smallest int8 or
     * greater than largest int8).
     *
     * Counterpart to Solidity's `int8` operator.
     *
     * Requirements:
     *
     * - input must fit into 8 bits
     */
    function toInt8(int256 value) internal pure returns (int8 downcasted) {
        downcasted = int8(value);
        if (downcasted != value) {
            revert SafeCastOverflowedIntDowncast(8, value);
        }
    }

    /**
     * @dev Converts an unsigned uint256 into a signed int256.
     *
     * Requirements:
     *
     * - input must be less than or equal to maxInt256.
     */
    function toInt256(uint256 value) internal pure returns (int256) {
        // Note: Unsafe cast below is okay because `type(int256).max` is guaranteed to be positive
        if (value > uint256(type(int256).max)) {
            revert SafeCastOverflowedUintToInt(value);
        }
        return int256(value);
    }

    /**
     * @dev Cast a boolean (false or true) to a uint256 (0 or 1) with no jump.
     */
    function toUint(bool b) internal pure returns (uint256 u) {
        assembly ("memory-safe") {
            u := iszero(iszero(b))
        }
    }
}
