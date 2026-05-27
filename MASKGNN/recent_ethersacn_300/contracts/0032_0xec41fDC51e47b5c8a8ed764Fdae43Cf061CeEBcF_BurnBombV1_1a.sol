// ===== FILE: npm/_chainlink/contracts_1.5.0/src/v0.8/vrf/dev/interfaces/IVRFCoordinatorV2Plus.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import {VRFV2PlusClient} from "../libraries/VRFV2PlusClient.sol";
import {IVRFSubscriptionV2Plus} from "./IVRFSubscriptionV2Plus.sol";

// Interface that enables consumers of VRFCoordinatorV2Plus to be future-proof for upgrades
// This interface is supported by subsequent versions of VRFCoordinatorV2Plus
interface IVRFCoordinatorV2Plus is IVRFSubscriptionV2Plus {
  /**
   * @notice Request a set of random words.
   * @param req - a struct containing following fields for randomness request:
   * keyHash - Corresponds to a particular oracle job which uses
   * that key for generating the VRF proof. Different keyHash's have different gas price
   * ceilings, so you can select a specific one to bound your maximum per request cost.
   * subId  - The ID of the VRF subscription. Must be funded
   * with the minimum subscription balance required for the selected keyHash.
   * requestConfirmations - How many blocks you'd like the
   * oracle to wait before responding to the request. See SECURITY CONSIDERATIONS
   * for why you may want to request more. The acceptable range is
   * [minimumRequestBlockConfirmations, 200].
   * callbackGasLimit - How much gas you'd like to receive in your
   * fulfillRandomWords callback. Note that gasleft() inside fulfillRandomWords
   * may be slightly less than this amount because of gas used calling the function
   * (argument decoding etc.), so you may need to request slightly more than you expect
   * to have inside fulfillRandomWords. The acceptable range is
   * [0, maxGasLimit]
   * numWords - The number of uint256 random values you'd like to receive
   * in your fulfillRandomWords callback. Note these numbers are expanded in a
   * secure way by the VRFCoordinator from a single random value supplied by the oracle.
   * extraArgs - abi-encoded extra args
   * @return requestId - A unique identifier of the request. Can be used to match
   * a request to a response in fulfillRandomWords.
   */
  function requestRandomWords(
    VRFV2PlusClient.RandomWordsRequest calldata req
  ) external returns (uint256 requestId);
}


// ===== FILE: npm/_chainlink/contracts_1.5.0/src/v0.8/vrf/dev/interfaces/IVRFSubscriptionV2Plus.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @notice The IVRFSubscriptionV2Plus interface defines the subscription
/// @notice related methods implemented by the V2Plus coordinator.
interface IVRFSubscriptionV2Plus {
  /**
   * @notice Add a consumer to a VRF subscription.
   * @param subId - ID of the subscription
   * @param consumer - New consumer which can use the subscription
   */
  function addConsumer(uint256 subId, address consumer) external;

  /**
   * @notice Remove a consumer from a VRF subscription.
   * @param subId - ID of the subscription
   * @param consumer - Consumer to remove from the subscription
   */
  function removeConsumer(uint256 subId, address consumer) external;

  /**
   * @notice Cancel a subscription
   * @param subId - ID of the subscription
   * @param to - Where to send the remaining LINK to
   */
  function cancelSubscription(uint256 subId, address to) external;

  /**
   * @notice Accept subscription owner transfer.
   * @param subId - ID of the subscription
   * @dev will revert if original owner of subId has
   * not requested that msg.sender become the new owner.
   */
  function acceptSubscriptionOwnerTransfer(
    uint256 subId
  ) external;

  /**
   * @notice Request subscription owner transfer.
   * @param subId - ID of the subscription
   * @param newOwner - proposed new owner of the subscription
   */
  function requestSubscriptionOwnerTransfer(uint256 subId, address newOwner) external;

  /**
   * @notice Create a VRF subscription.
   * @return subId - A unique subscription id.
   * @dev You can manage the consumer set dynamically with addConsumer/removeConsumer.
   * @dev Note to fund the subscription with LINK, use transferAndCall. For example
   * @dev  LINKTOKEN.transferAndCall(
   * @dev    address(COORDINATOR),
   * @dev    amount,
   * @dev    abi.encode(subId));
   * @dev Note to fund the subscription with Native, use fundSubscriptionWithNative. Be sure
   * @dev  to send Native with the call, for example:
   * @dev COORDINATOR.fundSubscriptionWithNative{value: amount}(subId);
   */
  function createSubscription() external returns (uint256 subId);

  /**
   * @notice Get a VRF subscription.
   * @param subId - ID of the subscription
   * @return balance - LINK balance of the subscription in juels.
   * @return nativeBalance - native balance of the subscription in wei.
   * @return reqCount - Requests count of subscription.
   * @return owner - owner of the subscription.
   * @return consumers - list of consumer address which are able to use this subscription.
   */
  function getSubscription(
    uint256 subId
  )
    external
    view
    returns (uint96 balance, uint96 nativeBalance, uint64 reqCount, address owner, address[] memory consumers);

  /*
   * @notice Check to see if there exists a request commitment consumers
   * for all consumers and keyhashes for a given sub.
   * @param subId - ID of the subscription
   * @return true if there exists at least one unfulfilled request for the subscription, false
   * otherwise.
   */
  function pendingRequestExists(
    uint256 subId
  ) external view returns (bool);

  /**
   * @notice Paginate through all active VRF subscriptions.
   * @param startIndex index of the subscription to start from
   * @param maxCount maximum number of subscriptions to return, 0 to return all
   * @dev the order of IDs in the list is **not guaranteed**, therefore, if making successive calls, one
   * @dev should consider keeping the blockheight constant to ensure a holistic picture of the contract state
   */
  function getActiveSubscriptionIds(uint256 startIndex, uint256 maxCount) external view returns (uint256[] memory);

  /**
   * @notice Fund a subscription with native.
   * @param subId - ID of the subscription
   * @notice This method expects msg.value to be greater than or equal to 0.
   */
  function fundSubscriptionWithNative(
    uint256 subId
  ) external payable;
}


// ===== FILE: npm/_chainlink/contracts_1.5.0/src/v0.8/vrf/dev/libraries/VRFV2PlusClient.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

// End consumer library.
library VRFV2PlusClient {
  // extraArgs will evolve to support new features
  bytes4 public constant EXTRA_ARGS_V1_TAG = bytes4(keccak256("VRF ExtraArgsV1"));

  struct ExtraArgsV1 {
    bool nativePayment;
  }

  struct RandomWordsRequest {
    bytes32 keyHash;
    uint256 subId;
    uint16 requestConfirmations;
    uint32 callbackGasLimit;
    uint32 numWords;
    bytes extraArgs;
  }

  function _argsToBytes(
    ExtraArgsV1 memory extraArgs
  ) internal pure returns (bytes memory bts) {
    return abi.encodeWithSelector(EXTRA_ARGS_V1_TAG, extraArgs);
  }
}


// ===== FILE: npm/_openzeppelin/contracts_4.8.3/access/Ownable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.7.0) (access/Ownable.sol)

pragma solidity ^0.8.0;

import "../utils/Context.sol";

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Initializes the contract setting the deployer as the initial owner.
     */
    constructor() {
        _transferOwnership(_msgSender());
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions anymore. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby removing any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}


// ===== FILE: npm/_openzeppelin/contracts_4.8.3/access/Ownable2Step.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.8.0) (access/Ownable2Step.sol)

pragma solidity ^0.8.0;

import "./Ownable.sol";

/**
 * @dev Contract module which provides access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * By default, the owner account will be the one that deploys the contract. This
 * can later be changed with {transferOwnership} and {acceptOwnership}.
 *
 * This module is used through inheritance. It will make available all functions
 * from parent (Ownable).
 */
abstract contract Ownable2Step is Ownable {
    address private _pendingOwner;

    event OwnershipTransferStarted(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev Returns the address of the pending owner.
     */
    function pendingOwner() public view virtual returns (address) {
        return _pendingOwner;
    }

    /**
     * @dev Starts the ownership transfer of the contract to a new account. Replaces the pending transfer if there is one.
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual override onlyOwner {
        _pendingOwner = newOwner;
        emit OwnershipTransferStarted(owner(), newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`) and deletes any pending owner.
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual override {
        delete _pendingOwner;
        super._transferOwnership(newOwner);
    }

    /**
     * @dev The new owner accepts the ownership transfer.
     */
    function acceptOwnership() external {
        address sender = _msgSender();
        require(pendingOwner() == sender, "Ownable2Step: caller is not the new owner");
        _transferOwnership(sender);
    }
}


// ===== FILE: npm/_openzeppelin/contracts_4.8.3/security/Pausable.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.7.0) (security/Pausable.sol)

pragma solidity ^0.8.0;

import "../utils/Context.sol";

/**
 * @dev Contract module which allows children to implement an emergency stop
 * mechanism that can be triggered by an authorized account.
 *
 * This module is used through inheritance. It will make available the
 * modifiers `whenNotPaused` and `whenPaused`, which can be applied to
 * the functions of your contract. Note that they will not be pausable by
 * simply including this module, only once the modifiers are put in place.
 */
abstract contract Pausable is Context {
    /**
     * @dev Emitted when the pause is triggered by `account`.
     */
    event Paused(address account);

    /**
     * @dev Emitted when the pause is lifted by `account`.
     */
    event Unpaused(address account);

    bool private _paused;

    /**
     * @dev Initializes the contract in unpaused state.
     */
    constructor() {
        _paused = false;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is not paused.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    modifier whenNotPaused() {
        _requireNotPaused();
        _;
    }

    /**
     * @dev Modifier to make a function callable only when the contract is paused.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    modifier whenPaused() {
        _requirePaused();
        _;
    }

    /**
     * @dev Returns true if the contract is paused, and false otherwise.
     */
    function paused() public view virtual returns (bool) {
        return _paused;
    }

    /**
     * @dev Throws if the contract is paused.
     */
    function _requireNotPaused() internal view virtual {
        require(!paused(), "Pausable: paused");
    }

    /**
     * @dev Throws if the contract is not paused.
     */
    function _requirePaused() internal view virtual {
        require(paused(), "Pausable: not paused");
    }

    /**
     * @dev Triggers stopped state.
     *
     * Requirements:
     *
     * - The contract must not be paused.
     */
    function _pause() internal virtual whenNotPaused {
        _paused = true;
        emit Paused(_msgSender());
    }

    /**
     * @dev Returns to normal state.
     *
     * Requirements:
     *
     * - The contract must be paused.
     */
    function _unpause() internal virtual whenPaused {
        _paused = false;
        emit Unpaused(_msgSender());
    }
}


// ===== FILE: npm/_openzeppelin/contracts_4.8.3/security/ReentrancyGuard.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v4.8.0) (security/ReentrancyGuard.sol)

pragma solidity ^0.8.0;

/**
 * @dev Contract module that helps prevent reentrant calls to a function.
 *
 * Inheriting from `ReentrancyGuard` will make the {nonReentrant} modifier
 * available, which can be applied to functions to make sure there are no nested
 * (reentrant) calls to them.
 *
 * Note that because there is a single `nonReentrant` guard, functions marked as
 * `nonReentrant` may not call one another. This can be worked around by making
 * those functions `private`, and then adding `external` `nonReentrant` entry
 * points to them.
 *
 * TIP: If you would like to learn more about reentrancy and alternative ways
 * to protect against it, check out our blog post
 * https://blog.openzeppelin.com/reentrancy-after-istanbul/[Reentrancy After Istanbul].
 */
abstract contract ReentrancyGuard {
    // Booleans are more expensive than uint256 or any type that takes up a full
    // word because each write operation emits an extra SLOAD to first read the
    // slot's contents, replace the bits taken up by the boolean, and then write
    // back. This is the compiler's defense against contract upgrades and
    // pointer aliasing, and it cannot be disabled.

    // The values being non-zero value makes deployment a bit more expensive,
    // but in exchange the refund on every call to nonReentrant will be lower in
    // amount. Since refunds are capped to a percentage of the total
    // transaction's gas, it is best to keep them low in cases like this one, to
    // increase the likelihood of the full refund coming into effect.
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

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
        // On the first call to nonReentrant, _status will be _NOT_ENTERED
        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        // Any calls to nonReentrant after this point will fail
        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {
        // By storing the original value once again, a refund is triggered (see
        // https://eips.ethereum.org/EIPS/eip-2200)
        _status = _NOT_ENTERED;
    }
}


// ===== FILE: npm/_openzeppelin/contracts_4.8.3/utils/Context.sol =====
// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (utils/Context.sol)

pragma solidity ^0.8.0;

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
}


// ===== FILE: project/src/BurnBombV1_1a.sol =====
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

// ─────────────────────────────────────────────────────────────────────────────
//  BurnBomb · Minnow v1.1a
//
//  Pre-public-launch redeploy of Minnow v1.1 (which was deployed + activated
//  2026-05-24 at 0x555b80b8d7c8c68e670f22c1b7f3a33e0b60d6ea but never went
//  public — gated to founder wallets via frontend env-var). v1.1a bundles
//  three additions before public launch:
//
//    §A  Throne royalty mechanic (3-seat creator leaderboard).
//        - WINNER_BPS 7250 → 7150; THRONE_BPS = 100 (1% of pot).
//        - Split 0.7 / 0.2 / 0.1 across seats 1 / 2 / 3.
//        - One-seat-per-creator rule (your biggest pot wins your seat).
//        - Throne UPDATE happens BEFORE throne PAYOUT in finalizeSettlement
//          so new ascenders earn their first throne payout on their
//          crowning round (operator-locked dramatic-payout design).
//        - 90-day season auto-reset; clock starts at activate().
//        - Empty throne seats overflow back to winner (matches v1.1's
//          VRF-cap-overflow pattern).
//    §B  On-chain Round.name (bytes32) + Round.imageURI (string).
//        - Fixes the v1.1 metadata oversight (creators' branding lived
//          off-chain only).
//        - Permanent + indexable by anyone; no backend dependency.
//    §C  Drop createRound() gas-only variants; only createRoundAndEnter
//        remains as the canonical opener (skin-in-the-game enforced at
//        the contract level).
//
//  Plus: Round.creator field added (needed for throne lookup; the address
//  was already emitted in RoundCreated but not stored on the struct).
//
//  UNCHANGED from v1.1 (verified preserved):
//    - VRF v2.5 + IMMUTABLE s_vrfCoordinator (no setCoordinator bypass)
//    - renounceOwnership() reverts unconditionally
//    - Single-active-round invariant
//    - Bulk withdraw functions (refund / payout / seed dividend)
//    - cancelVrfSubscription (deprecated + terminal + no-pending gates)
//    - freezeAdmin + adminNotFrozen on pause/unpause/cancelVrfSubscription
//    - Per-round tunable threshold (MIN/DEFAULT/MAX_THRESHOLD)
//    - VRF cap with overflow-to-winner
//    - 72h emergency refund path
//    - Pull-payment everywhere
//    - Two-step founder rotation (proposeFounderPayoutAddress + accept)
//    - Ownable2Step from OpenZeppelin (NOT Chainlink ConfirmedOwner)
//    - VRFConsumerBaseV2Plus NOT inherited (inline rawFulfillRandomWords)
//
//  Storage layout: independent from v1 + v1.1 (new deploy, not a proxy
//  upgrade). v1.1 (0x555b80b8...) will be deprecated as part of v1.1a
//  deploy; 0.05 Ξ VRF reserve recovered via cancelVrfSubscription.
//
//  See: burnbomb/MINNOW-V1.1A-SPEC.md  (canonical, this deploy)
//       burnbomb/MINNOW-V1.1-SPEC.md   (baseline reference)
//       contracts/src/BurnBombV1_1.sol  (v1.1 predecessor — fork base)
//       memory: project_burnbomb_v1_1a_throne_bundle.md
// ─────────────────────────────────────────────────────────────────────────────

import {Ownable2Step} from "@openzeppelin/contracts/access/Ownable2Step.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/security/Pausable.sol";

import {VRFV2PlusClient} from "@chainlink/contracts/src/v0.8/vrf/dev/libraries/VRFV2PlusClient.sol";
import {IVRFCoordinatorV2Plus} from "@chainlink/contracts/src/v0.8/vrf/dev/interfaces/IVRFCoordinatorV2Plus.sol";

// v1.1a · External library extracted to fit EIP-170 24KB ceiling. ThroneLib
// owns the bubble-sort + one-seat-per-creator mutation logic; delegate-called
// from finalizeSettlement. ~1KB saved off main bytecode.
import {ThroneLib} from "./lib/ThroneLib.sol";

// ─── VRF consumer logic — INLINED in v1.1 instead of inheriting VRFConsumerBaseV2Plus ───
//
// v1 inherited Chainlink's `VRFConsumerBaseV2Plus`, which exposes
// `setCoordinator(address)` to `owner()` (and current coordinator). That function
// is NOT virtual (cannot be overridden) and NOT guarded by our `adminNotFrozen`
// modifier. After `freezeAdmin()`, the owner could still swap `s_vrfCoordinator`
// to an attacker-controlled address that then injects chosen randomness via
// `rawFulfillRandomWords`. (Third Codex pass 2026-05-24 BLOCKER finding.)
//
// v1.1 fix: do NOT inherit the base. Inline only the bits we need:
//   - `s_vrfCoordinator` declared `immutable` (set once in constructor, can't change)
//   - `rawFulfillRandomWords` external dispatch with msg.sender == coordinator check
//   - `fulfillRandomWords` is now a plain `internal` virtual hook (no override)
//   - No `setCoordinator` surface exposed at all
//
// Ownership: switch from Chainlink's `ConfirmedOwner` (two-step transfer) to
// OpenZeppelin's `Ownable2Step` — same two-step UX, but decoupled from VRF.

contract BurnBombV1_1a is Ownable2Step, ReentrancyGuard, Pausable {
    // ─────────────────────────────────────────────────────────────────────
    //  Errors
    // ─────────────────────────────────────────────────────────────────────

    error BurnBomb_ZeroAddress();
    error WrongPhase();
    error WrongValue();
    error SlotTaken();
    error SlotOutOfRange();
    error NotSlotOwner();
    error AddressCapExceeded();
    error TxCapExceeded();
    error BatchValueMismatch();
    error EmptyBatch();
    error ActiveRoundExists();
    error NoActiveRound();
    error ContractDeprecated();
    error ContractNotDeprecated();
    error NotActivated();
    error AlreadyActivated();
    error AdminIsFrozen();
    error AlreadyFrozen();
    error AlreadyDeprecated();
    error NotProposedRecipient();
    error NoPendingProposal();
    error EmergencyTooEarly();
    error CannotFreezeWhilePaused();
    error InvalidConstructorParam();
    error VrfTooEarly();
    error VrfReserveTooLow();
    error VrfStillPending();
    error VrfBadCallback();
    error VrfStaleCallback();
    error VrfNotFulfilled();
    error VrfHealthCheckNotPassed();
    error VrfRetryCapReached();
    error VrfRequestPending();
    error ZeroValue();
    error NothingToWithdraw();
    error AlreadyClaimed();
    error DirectSendsRejected();
    error ThresholdOutOfRange(uint256 supplied, uint256 min, uint256 max);
    error ZeroWinnerOnOverflow();
    error BatchTooLarge();
    /// @dev VRF callback dispatch — only the immutable coordinator can fulfill.
    error OnlyCoordinatorCanFulfill(address have, address want);
    /// @dev v1.1a NEW · Round metadata length caps for name + imageURI.
    error NameTooLong(uint256 length, uint256 max);
    error ImageURITooLong(uint256 length, uint256 max);

    // ─────────────────────────────────────────────────────────────────────
    //  Constants (audit-fixed)
    // ─────────────────────────────────────────────────────────────────────

    uint16  public constant MAX_ENTRIES_PER_ADDRESS = 50;
    uint16  public constant MAX_ENTRIES_PER_TX = 10;
    uint256 public constant EMERGENCY_REFUND_DELAY = 72 hours;
    uint256 public constant VRF_RETRY_TIMEOUT = 30 minutes;
    uint256 public constant VRF_RETRY_BOUNTY = 0.0001 ether;
    uint256 public constant MAX_VRF_RETRY_BOUNTY_PER_ROUND = 0.005 ether;
    uint32  public constant MAX_CHAINLINK_REQUESTS_PER_ROUND = 3;

    // Pot split · v1.1a carves 100 BPS from WINNER (was 7250) for THRONE.
    //   v1.1   : WINNER=7250, ..., total 10000 (no THRONE slice)
    //   v1.1a  : WINNER=7150, THRONE=100, ..., total 10000
    // See MINNOW-V1.1A-SPEC.md §4 for the BPS conservation proof + winnerShare
    // worst-case underflow check at MIN_THRESHOLD (winnerShare ≈ 0.00577 Ξ).
    uint16  public constant WINNER_BPS  = 7150;  // v1.1a: was 7250 in v1.1
    uint16  public constant SECOND_BPS  = 650;
    uint16  public constant THIRD_BPS   = 350;
    uint16  public constant SEED_BPS    = 1000;
    uint16  public constant RAKE_BPS    = 500;
    uint16  public constant FOUNDER_BPS = 50;
    uint16  public constant KEEPER_BPS  = 100;
    uint16  public constant VRF_REPLENISH_BPS = 100;
    uint256 public constant KEEPER_CAP = 0.01 ether;
    uint16  public constant BPS_DENOM  = 10000;

    uint256 public constant MIN_VRF_RESERVE_TO_ACTIVATE = 0.025 ether;

    address public constant BURN_TOKEN = 0x886fab7097311af73C74DD608001a0d267AbF351;

    address public constant VRF_COORDINATOR_MAINNET = 0xD7f86b4b8Cae7D942340FF628F82735b7a20893a;
    uint16  public constant VRF_REQUEST_CONFIRMATIONS = 3;
    uint32  public constant VRF_CALLBACK_GAS_LIMIT = 500000;
    uint32  public constant VRF_NUM_WORDS = 1;

    // ── v1.1 new constants (per spec §2.2, §3.2, §5.5) ──

    /// @notice Minimum allowed threshold for a round.
    /// Math floor for winnerShare-underflow safety:
    ///   MAX_VRF_RETRY_BOUNTY_PER_ROUND / (1 - SUM_OTHER_BPS/BPS_DENOM)
    ///   = 0.005 / 0.725 ≈ 0.0069 ETH.
    /// 0.015 gives ~2.17x headroom. (sub-agent audit 2026-05-24)
    uint256 public constant MIN_THRESHOLD = 0.015 ether;

    /// @notice Default threshold for the no-arg createRound() overload.
    /// Operator-chosen (2026-05-24) at 0.05 ETH to encourage meaningful
    /// seed commitment while remaining tunable down to MIN_THRESHOLD.
    uint256 public constant DEFAULT_THRESHOLD = 0.05 ether;

    /// @notice Maximum allowed threshold for a round.
    /// Conservative ceiling against unreachable thresholds.
    uint256 public constant MAX_THRESHOLD = 1 ether;

    /// @notice VRF sub balance cap. At or above this, the per-round VRF
    /// replenish slice flows to the round winner instead of to Chainlink.
    uint256 public constant MAX_VRF_RESERVE_BALANCE = 0.05 ether;

    /// @notice Maximum array length for any batch withdraw call.
    /// Bracket-independent — NOT derived from NUM_SLOTS. Future brackets
    /// (Dolphin 1000-slot etc.) keep this at ~100 for gas safety regardless
    /// of their NUM_SLOTS. Do not refactor as shared-with-NUM_SLOTS constant.
    /// (sub-agent audit 2026-05-24)
    uint256 public constant MAX_BATCH_SIZE = 100;

    // ── v1.1a NEW · Throne royalty mechanic ──
    // See MINNOW-V1.1A-SPEC.md §2 for full rationale. Top-3 creators ranked
    // by largest-ever round pot earn 1% of every future pot (split 70/20/10).
    // 90-day seasons. One-seat-per-creator (biggest pot wins their seat).
    // Throne UPDATE happens BEFORE PAYOUT — new ascenders earn their first
    // payout on their crowning round (operator-locked dramatic design).

    /// @notice 1% of pot carved from WINNER_BPS for the throne pool.
    uint16  public constant THRONE_BPS = 100;
    /// @notice Throne seat split — MUST sum to 100. Encoded as separate
    /// constants for audit clarity vs an array index lookup.
    uint16  public constant THRONE_SEAT_1_BPS_OF_THRONE = 70;
    uint16  public constant THRONE_SEAT_2_BPS_OF_THRONE = 20;
    uint16  public constant THRONE_SEAT_3_BPS_OF_THRONE = 10;
    /// @notice Season duration. After this elapsed since the last reset
    /// (anchored at activate()), the next finalizeSettlement wipes the
    /// throne to empty and re-anchors the timer.
    uint256 public constant SEASON_LENGTH = 90 days;

    // ── v1.1a NEW · Round metadata length caps ──
    /// @notice Max bytes for Round.name (UTF-8). Frontend's NAME_MAX=32
    /// reflects this cap. Multi-byte chars (emoji = 4 bytes) reduce
    /// effective char count.
    uint256 public constant MAX_NAME_BYTES = 32;
    /// @notice Max bytes for Round.imageURI. Fits IPFS CIDs (~46 chars),
    /// most HTTPS URLs, and Arweave URIs. Caps gas griefing potential.
    uint256 public constant MAX_IMAGE_URI_BYTES = 256;

    // ─────────────────────────────────────────────────────────────────────
    //  Immutables (bracket parameters set at deploy)
    // ─────────────────────────────────────────────────────────────────────

    uint256 public immutable MIN_SLOT;
    uint256 public immutable SLOT_STEP;
    uint16  public immutable NUM_SLOTS;
    // NOTE: v1's `uint256 public immutable THRESHOLD;` REMOVED in v1.1.
    // Threshold is per-round, lives on Round.threshold (set in createRound).
    // See spec §2.7 CRITICAL IMPLEMENTATION DIRECTIVE.

    uint256 public immutable SEED_PHASE_MAX;
    uint256 public immutable ACTIVE_TIMER;
    uint256 public immutable ACTIVE_PHASE_MAX;
    uint256 public immutable FINAL_COUNTDOWN;

    bytes32 public immutable VRF_KEY_HASH;
    uint256 public immutable VRF_SUB_ID;

    /// @notice Chainlink VRF coordinator. IMMUTABLE in v1.1 — cannot be swapped
    /// post-deploy. (v1 inherited a mutable `s_vrfCoordinator` from
    /// VRFConsumerBaseV2Plus exposed via `setCoordinator()`; v1.1 removes that
    /// attack surface entirely. Codex 2026-05-24.)
    IVRFCoordinatorV2Plus public immutable s_vrfCoordinator;

    // ─────────────────────────────────────────────────────────────────────
    //  Storage
    // ─────────────────────────────────────────────────────────────────────

    enum Phase {
        NONE,
        SEEDING,
        ACTIVE,
        FINAL_COUNTDOWN,
        READY_TO_REQUEST_RANDOMNESS,
        WAITING_FOR_RANDOMNESS,
        RANDOMNESS_FULFILLED,
        SETTLED,
        REFUNDED
    }

    struct Round {
        // --- timing packed in 1 slot ---
        uint64 createdAt;
        uint64 activeStartedAt;
        uint64 activeTimerEndsAt;
        uint64 finalCountdownEndsAt;

        // --- timing + state packed in 1 slot ---
        uint64 seedEndsAt;
        uint64 activeEndsAt;
        uint16 slotsClaimed;
        uint8  phase;
        uint32 retryCount;

        // --- pot accounting ---
        uint256 potTotal;
        uint256 seedPotTotal;

        // --- pot split snapshot ---
        uint256 seedDividendPool;

        // --- VRF state ---
        uint256 vrfRequestId;
        uint256 vrfRequestedAt;
        uint256 firstVrfRequestedAt;
        uint256 randomWord;
        uint256 totalRetryBountyPaid;

        // --- Best-effort BURN rake escrow ---
        uint256 pendingBurnRake;

        // --- finalization snapshots ---
        uint16  winnerSlot;
        uint16  secondSlot;
        uint16  thirdSlot;
        address founderPayoutAtSettle;

        // --- v1.1 NEW: per-round seed-cross threshold (wei) ---
        // MUST be the final field of Round per spec §8 (defensive storage
        // layout — append-only, never insert between existing fields).
        uint256 threshold;

        // --- v1.1a NEW: append-only (storage layout safety) ---
        // creator: who called createRoundAndEnter; needed for throne lookup.
        //   IMMUTABLE post-creation — no setter exists. Set once in
        //   createRoundAndEnter; read-only thereafter.
        // name: UTF-8 round name. On-chain length cap MAX_NAME_BYTES.
        // imageURI: ipfs:// | https:// | ar:// pointer. Empty for default.
        //   On-chain length cap MAX_IMAGE_URI_BYTES.
        address creator;
        string  name;
        string  imageURI;
    }

    uint256 public currentRoundId;
    mapping(uint256 => Round) public rounds;

    mapping(uint256 => mapping(uint16 => address)) public slotOwner;
    mapping(uint256 => uint16[]) private _claimedSlots;

    mapping(uint256 => mapping(address => uint16)) public addressEntryCount;
    mapping(uint256 => mapping(address => uint16[])) public addressSlots;

    mapping(uint256 => mapping(address => uint256)) public seedContribution;
    mapping(uint256 => mapping(address => bool)) public seedDividendClaimed;

    mapping(uint256 => mapping(address => uint256)) public claimablePayout;

    mapping(uint256 => uint256) public vrfRequestIdToRoundId;

    uint256 public vrfHealthRequestId;
    uint256 public vrfHealthRequestedAt;
    uint256 public lastVrfHealthFulfilledAt;
    uint256 public lastVrfHealthRandomWord;
    bool    public vrfHealthCheckPassed;

    bool public adminFrozen;
    bool public deprecated;
    bool public activated;

    address public founderPayout;
    address public pendingFounderPayout;

    // ── v1.1a NEW · Throne royalty state ──
    // throne[0] = seat 1 (highest); throne[2] = seat 3 (lowest). Empty seats
    // are address(0) with throneSize 0. Sorted descending by throneSize with
    // ties favoring incumbent. One-seat-per-creator enforced in _updateThrone.
    address[3] public throne;
    uint256[3] public throneSize;

    /// @notice 90-day season anchor. Set in activate() so the clock aligns
    /// with operational launch (NOT deploy time). Reset in finalizeSettlement
    /// when block.timestamp >= seasonStartedAt + SEASON_LENGTH.
    uint64 public seasonStartedAt;

    // ─────────────────────────────────────────────────────────────────────
    //  Events
    // ─────────────────────────────────────────────────────────────────────

    // v1.1 SIGNATURE CHANGE: appends `threshold` so off-chain consumers see
    // the per-round value without an extra read.
    // v1.1a SIGNATURE CHANGE: adds `name` + `imageURI` so off-chain consumers
    // can render branded rounds from the event stream alone (no contract read needed).
    event RoundCreated(
        uint256 indexed roundId,
        address indexed creator,
        uint256 createdAt,
        uint256 threshold,
        string name,
        string imageURI
    );
    // v1.1a NEW · Throne event surface (size-optimized per Codex audit)
    /// @notice Emitted ONCE per finalizeSettlement with the deterministic
    /// post-state snapshot of the throne. Indexers compute per-seat diffs
    /// by comparing successive ThroneSnapshot events from the same round-id
    /// stream — cleaner than reconstructing from interleaved per-seat events.
    event ThroneSnapshot(
        uint256 indexed roundId,
        address seat1Owner,
        uint256 seat1Size,
        address seat2Owner,
        uint256 seat2Size,
        address seat3Owner,
        uint256 seat3Size
    );
    /// @notice Emitted when the 90-day season auto-resets in finalizeSettlement.
    /// The accompanying ThroneSnapshot in the same tx shows the wiped (zeroed) state.
    event SeasonReset(uint64 oldSeasonStart, uint64 newSeasonStart);

    event SlotClaimed(uint256 indexed roundId, uint16 indexed slotIndex, address indexed claimer, uint256 value);
    event PhaseTransition(uint256 indexed roundId, Phase oldPhase, Phase newPhase);
    event ThresholdMet(uint256 indexed roundId, uint256 potAtThreshold);
    event FinalCountdownStarted(uint256 indexed roundId);
    event ReadyToRequestRandomness(uint256 indexed roundId);

    event RandomnessRequested(uint256 indexed roundId, uint256 indexed requestId);
    event RandomnessFulfilled(uint256 indexed roundId, uint256 indexed requestId, uint256 randomWord);
    event RandomnessRetried(uint256 indexed roundId, address indexed caller, uint256 indexed newRequestId, uint256 bountyPaid);
    event VrfReserveToppedUp(address indexed funder, uint256 amount);
    event VrfReserveReplenished(uint256 indexed roundId, uint256 amount);
    event VrfHealthCheckRequested(uint256 indexed requestId);
    event VrfHealthCheckFulfilled(uint256 indexed requestId, uint256 randomWord);
    event BurnRakeForwardDeferred(uint256 indexed roundId, uint256 amount);
    event BurnRakeForwarded(uint256 indexed roundId, uint256 amount);

    // v1.1 NEW: VRF replenish slice flowed (partially or fully) to winner
    // because the shared sub was at/above MAX_VRF_RESERVE_BALANCE.
    event VrfReplenishOverflowToWinner(uint256 indexed roundId, address indexed winner, uint256 amount);

    // v1.1 NEW: sub balance recovered on retire.
    event VrfSubscriptionCancelled(address indexed recipient, uint256 amountRecovered);

    event RoundSettled(
        uint256 indexed roundId,
        address indexed winner,
        uint16 winningSlot,
        uint16 target,
        uint256 pot,
        uint256 winnerShare,
        address secondPlace,
        uint256 secondShare,
        address thirdPlace,
        uint256 thirdShare,
        uint256 seedDividendPool,
        uint256 burnRake,
        address founder,
        uint256 founderShare,
        uint256 keeperBounty,
        address keeper
    );
    event RoundRefunded(uint256 indexed roundId);

    event PayoutWithdrawn(uint256 indexed roundId, address indexed account, address indexed recipient, uint256 amount);
    event SeedDividendWithdrawn(uint256 indexed roundId, address indexed account, uint256 amount);
    event RefundClaimed(uint256 indexed roundId, address indexed account, uint16 indexed slotIndex, uint256 amount);

    event FounderPayoutProposed(address indexed current, address indexed proposed);
    event FounderPayoutAccepted(address indexed old, address indexed accepted);

    event EmergencyRefundEnabled(uint256 indexed roundId);
    event AdminFrozen(address indexed owner);
    event Deprecated(address indexed owner, uint256 timestamp);
    event Activated(address indexed owner, uint256 timestamp);

    // ─────────────────────────────────────────────────────────────────────
    //  Modifiers
    // ─────────────────────────────────────────────────────────────────────

    modifier adminNotFrozen() {
        if (adminFrozen) revert AdminIsFrozen();
        _;
    }

    modifier notDeprecated() {
        if (deprecated) revert ContractDeprecated();
        _;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Constructor
    //
    //  v1.1 CHANGE: _threshold parameter REMOVED (per spec §2.7).
    //  Threshold is now per-round; bracket-level safety is provided by
    //  MIN_THRESHOLD/MAX_THRESHOLD constants enforced in createRound.
    //  v1's `if (_threshold < 10 * MAX_VRF_RETRY_BOUNTY_PER_ROUND) ...`
    //  safety check is also removed — equivalent safety lives in the
    //  ThresholdOutOfRange check at round-creation time.
    // ─────────────────────────────────────────────────────────────────────

    constructor(
        address _vrfCoordinator,
        bytes32 _vrfKeyHash,
        uint256 _minSlot,
        uint256 _slotStep,
        uint16 _numSlots,
        uint256 _seedPhaseMax,
        uint256 _activeTimer,
        uint256 _activePhaseMax,
        uint256 _finalCountdown,
        address _founderPayout
    ) {
        if (_vrfCoordinator == address(0)) revert BurnBomb_ZeroAddress();
        if (_founderPayout == address(0)) revert BurnBomb_ZeroAddress();
        if (_vrfKeyHash == bytes32(0)) revert InvalidConstructorParam();

        if (_minSlot == 0) revert InvalidConstructorParam();
        if (_slotStep == 0) revert InvalidConstructorParam();
        if (_numSlots < 3 || _numSlots > 10000) revert InvalidConstructorParam();
        if (_seedPhaseMax == 0) revert InvalidConstructorParam();
        if (_activeTimer == 0) revert InvalidConstructorParam();
        if (_activePhaseMax < _activeTimer) revert InvalidConstructorParam();
        if (_finalCountdown == 0) revert InvalidConstructorParam();

        // MAX_THRESHOLD must be reachable: sum of all slots at face value must
        // be ≥ MAX_THRESHOLD. Otherwise a creator could pick a threshold that
        // can never be crossed even if every slot fills.
        //   sum of slots = MIN_SLOT × N + SLOT_STEP × (N×(N-1)/2)
        uint256 maxPossiblePot =
            _minSlot * uint256(_numSlots) + _slotStep * (uint256(_numSlots) * (uint256(_numSlots) - 1) / 2);
        if (MAX_THRESHOLD > maxPossiblePot) revert InvalidConstructorParam();

        VRF_KEY_HASH = _vrfKeyHash;
        MIN_SLOT = _minSlot;
        SLOT_STEP = _slotStep;
        NUM_SLOTS = _numSlots;
        SEED_PHASE_MAX = _seedPhaseMax;
        ACTIVE_TIMER = _activeTimer;
        ACTIVE_PHASE_MAX = _activePhaseMax;
        FINAL_COUNTDOWN = _finalCountdown;

        founderPayout = _founderPayout;

        // v1.1: bind coordinator IMMUTABLY — no setCoordinator surface (Codex 2026-05-24).
        s_vrfCoordinator = IVRFCoordinatorV2Plus(_vrfCoordinator);

        // Self-own VRF subscription
        VRF_SUB_ID = s_vrfCoordinator.createSubscription();
        s_vrfCoordinator.addConsumer(VRF_SUB_ID, address(this));
    }

    // ─────────────────────────────────────────────────────────────────────
    //  VRF callback dispatch · INLINED (replaces VRFConsumerBaseV2Plus)
    //
    //  rawFulfillRandomWords is the entry point Chainlink coordinators call
    //  to deliver fulfilled randomness. Gate it by msg.sender == the IMMUTABLE
    //  s_vrfCoordinator so the consumer accepts callbacks ONLY from the
    //  coordinator bound at deploy. No setCoordinator() surface means there
    //  is no admin path to swap the trusted source post-deploy — closes the
    //  freezeAdmin-bypass vector inherited from VRFConsumerBaseV2Plus in v1.
    // ─────────────────────────────────────────────────────────────────────

    function rawFulfillRandomWords(uint256 requestId, uint256[] calldata randomWords) external {
        if (msg.sender != address(s_vrfCoordinator)) {
            revert OnlyCoordinatorCanFulfill(msg.sender, address(s_vrfCoordinator));
        }
        fulfillRandomWords(requestId, randomWords);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Pricing helpers
    // ─────────────────────────────────────────────────────────────────────

    function slotValue(uint16 slotIndex) public view returns (uint256) {
        return MIN_SLOT + uint256(slotIndex) * SLOT_STEP;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Round lifecycle
    // ─────────────────────────────────────────────────────────────────────

    /// @notice v1.1a: ONLY entry point for opening a round. Creator commits
    /// a slot of their choosing in the same transaction, paying its face value.
    /// Skin-in-the-game enforced at the contract level (v1.1's gas-only
    /// `createRound()` variants are REMOVED in v1.1a per spec §5).
    /// @param threshold      Must be in [MIN_THRESHOLD, MAX_THRESHOLD].
    /// @param slotIdx        Slot index to claim. msg.value must equal slotValue(slotIdx).
    /// @param name           UTF-8 round name; max MAX_NAME_BYTES (32). Empty = default rendering.
    /// @param imageURI       Image URI (ipfs/https/ar); max MAX_IMAGE_URI_BYTES (256). Empty = default.
    function createRoundAndEnter(
        uint256 threshold,
        uint16 slotIdx,
        string calldata name,
        string calldata imageURI
    )
        external
        payable
        whenNotPaused
        returns (uint256 roundId)
    {
        if (!activated) revert NotActivated();
        if (deprecated) revert ContractDeprecated();
        if (currentRoundId != 0) {
            Phase prev = Phase(rounds[currentRoundId].phase);
            if (prev != Phase.SETTLED && prev != Phase.REFUNDED) revert ActiveRoundExists();
        }
        // Metadata length caps first (cheap checks before threshold/slot)
        if (bytes(name).length > MAX_NAME_BYTES) {
            revert NameTooLong(bytes(name).length, MAX_NAME_BYTES);
        }
        if (bytes(imageURI).length > MAX_IMAGE_URI_BYTES) {
            revert ImageURITooLong(bytes(imageURI).length, MAX_IMAGE_URI_BYTES);
        }
        if (threshold < MIN_THRESHOLD || threshold > MAX_THRESHOLD) {
            revert ThresholdOutOfRange(threshold, MIN_THRESHOLD, MAX_THRESHOLD);
        }
        if (slotIdx >= NUM_SLOTS) revert SlotOutOfRange();
        uint256 requiredValue = slotValue(slotIdx);
        if (msg.value != requiredValue) revert WrongValue();

        roundId = currentRoundId + 1;
        currentRoundId = roundId;
        Round storage r = rounds[roundId];
        r.createdAt = uint64(block.timestamp);
        r.seedEndsAt = uint64(block.timestamp + SEED_PHASE_MAX);
        r.phase = uint8(Phase.SEEDING);
        r.threshold = threshold;
        r.creator = msg.sender;       // v1.1a · immutable post-creation
        r.name = name;
        r.imageURI = imageURI;

        emit RoundCreated(roundId, msg.sender, block.timestamp, threshold, name, imageURI);
        emit PhaseTransition(roundId, Phase.NONE, Phase.SEEDING);

        // Reuse the EXACT v1.1 entry path. _enterSingleInternal handles all
        // address-cap, slot-taken, claimed-slot-array, address-slots, and
        // seed-contribution accounting — and any same-tx auto-transition
        // (threshold-cross to ACTIVE, sold-out fast-track).
        _enterSingleInternal(roundId, slotIdx, requiredValue);
    }

    /// @notice Single-slot entry. msg.value must equal exact slot value.
    function enter(uint256 roundId, uint16 slotIndex)
        external
        payable
        nonReentrant
        whenNotPaused
    {
        _advancePhase(roundId);
        _enterSingle(roundId, slotIndex, msg.value);
        _advancePhase(roundId);
    }

    /// @notice Batch entry. msg.value must equal exact sum of slot values.
    function enterBatch(uint256 roundId, uint16[] calldata slotIndexes)
        external
        payable
        nonReentrant
        whenNotPaused
    {
        if (slotIndexes.length == 0) revert EmptyBatch();
        if (slotIndexes.length > MAX_ENTRIES_PER_TX) revert TxCapExceeded();

        _advancePhase(roundId);

        uint256 totalRequired = 0;
        for (uint256 i = 0; i < slotIndexes.length; i++) {
            uint16 idx = slotIndexes[i];
            uint256 v = slotValue(idx);
            totalRequired += v;
            _enterSingleInternal(roundId, idx, v);
        }

        if (msg.value != totalRequired) revert BatchValueMismatch();

        _advancePhase(roundId);
    }

    function _enterSingle(uint256 roundId, uint16 slotIndex, uint256 value) internal {
        uint256 v = slotValue(slotIndex);
        if (value != v) revert WrongValue();
        _enterSingleInternal(roundId, slotIndex, v);
    }

    function _enterSingleInternal(uint256 roundId, uint16 slotIndex, uint256 value) internal {
        Round storage r = rounds[roundId];
        Phase p = Phase(r.phase);
        if (p != Phase.SEEDING && p != Phase.ACTIVE && p != Phase.FINAL_COUNTDOWN) revert WrongPhase();

        if (slotIndex >= NUM_SLOTS) revert SlotOutOfRange();
        if (slotOwner[roundId][slotIndex] != address(0)) revert SlotTaken();
        if (addressEntryCount[roundId][msg.sender] >= MAX_ENTRIES_PER_ADDRESS) revert AddressCapExceeded();

        // Effects
        slotOwner[roundId][slotIndex] = msg.sender;
        _claimedSlots[roundId].push(slotIndex);
        addressSlots[roundId][msg.sender].push(slotIndex);
        addressEntryCount[roundId][msg.sender] += 1;

        // v1.1: per-round threshold lives on r.threshold (was: const THRESHOLD).
        uint256 potBefore = r.potTotal;
        uint256 potAfter = potBefore + value;
        if (potBefore < r.threshold) {
            uint256 seedRoom = r.threshold - potBefore;
            uint256 seedShare = value < seedRoom ? value : seedRoom;
            seedContribution[roundId][msg.sender] += seedShare;
            r.seedPotTotal += seedShare;
        }

        r.potTotal = potAfter;
        r.slotsClaimed += 1;

        if (p == Phase.ACTIVE) {
            r.activeTimerEndsAt = uint64(block.timestamp + ACTIVE_TIMER);
        }

        emit SlotClaimed(roundId, slotIndex, msg.sender, value);

        // v1.1: per-round threshold-crossing transition
        if (p == Phase.SEEDING && potAfter >= r.threshold) {
            r.phase = uint8(Phase.ACTIVE);
            r.activeStartedAt = uint64(block.timestamp);
            r.activeTimerEndsAt = uint64(block.timestamp + ACTIVE_TIMER);
            r.activeEndsAt = uint64(block.timestamp + ACTIVE_PHASE_MAX);
            emit ThresholdMet(roundId, potAfter);
            emit PhaseTransition(roundId, Phase.SEEDING, Phase.ACTIVE);
        }

        if (r.slotsClaimed == NUM_SLOTS) {
            Phase prev = Phase(r.phase);
            r.phase = uint8(Phase.READY_TO_REQUEST_RANDOMNESS);
            emit PhaseTransition(roundId, prev, Phase.READY_TO_REQUEST_RANDOMNESS);
            emit ReadyToRequestRandomness(roundId);
        }
    }

    /// @notice Permissionless phase-advance.
    function tick(uint256 roundId) external {
        _advancePhase(roundId);
    }

    function _advancePhase(uint256 roundId) internal {
        Round storage r = rounds[roundId];
        Phase p = Phase(r.phase);

        // SEEDING: timeout to REFUND if threshold not met
        if (p == Phase.SEEDING) {
            // v1.1: per-round threshold (was: const THRESHOLD)
            if (block.timestamp >= r.seedEndsAt && r.potTotal < r.threshold) {
                r.phase = uint8(Phase.REFUNDED);
                emit PhaseTransition(roundId, Phase.SEEDING, Phase.REFUNDED);
                emit RoundRefunded(roundId);
            }
            return;
        }

        // ACTIVE: sliding timer expiry or hard-ceiling hit → FINAL_COUNTDOWN
        if (p == Phase.ACTIVE) {
            bool slidingExpired = block.timestamp >= r.activeTimerEndsAt;
            bool ceilingHit = block.timestamp >= r.activeEndsAt;
            if (slidingExpired || ceilingHit) {
                r.phase = uint8(Phase.FINAL_COUNTDOWN);
                r.finalCountdownEndsAt = uint64(block.timestamp + FINAL_COUNTDOWN);
                emit PhaseTransition(roundId, Phase.ACTIVE, Phase.FINAL_COUNTDOWN);
                emit FinalCountdownStarted(roundId);
            }
            return;
        }

        if (p == Phase.FINAL_COUNTDOWN) {
            if (block.timestamp >= r.finalCountdownEndsAt) {
                r.phase = uint8(Phase.READY_TO_REQUEST_RANDOMNESS);
                emit PhaseTransition(roundId, Phase.FINAL_COUNTDOWN, Phase.READY_TO_REQUEST_RANDOMNESS);
                emit ReadyToRequestRandomness(roundId);
            }
            return;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Settlement
    // ─────────────────────────────────────────────────────────────────────

    function topUpVrfReserve() external payable {
        if (msg.value == 0) revert ZeroValue();
        s_vrfCoordinator.fundSubscriptionWithNative{value: msg.value}(VRF_SUB_ID);
        emit VrfReserveToppedUp(msg.sender, msg.value);
    }

    function requestVrfHealthCheck() external onlyOwner {
        if (activated) revert AlreadyActivated();
        if (vrfHealthRequestId != 0 && !vrfHealthCheckPassed) revert VrfStillPending();
        if (vrfReserveBalance() < MIN_VRF_RESERVE_TO_ACTIVATE) revert VrfReserveTooLow();

        uint256 requestId = _requestRandomWords();
        vrfHealthRequestId = requestId;
        vrfHealthRequestedAt = block.timestamp;

        emit VrfHealthCheckRequested(requestId);
    }

    function settle(uint256 roundId) external nonReentrant {
        _advancePhase(roundId);

        Round storage r = rounds[roundId];
        if (Phase(r.phase) != Phase.READY_TO_REQUEST_RANDOMNESS) revert WrongPhase();

        uint256 requestId = _requestRandomWords();

        r.vrfRequestId = requestId;
        r.vrfRequestedAt = block.timestamp;
        r.firstVrfRequestedAt = block.timestamp;
        r.phase = uint8(Phase.WAITING_FOR_RANDOMNESS);
        vrfRequestIdToRoundId[requestId] = roundId;

        emit PhaseTransition(roundId, Phase.READY_TO_REQUEST_RANDOMNESS, Phase.WAITING_FOR_RANDOMNESS);
        emit RandomnessRequested(roundId, requestId);
    }

    function _requestRandomWords() internal returns (uint256 requestId) {
        requestId = s_vrfCoordinator.requestRandomWords(
            VRFV2PlusClient.RandomWordsRequest({
                keyHash: VRF_KEY_HASH,
                subId: VRF_SUB_ID,
                requestConfirmations: VRF_REQUEST_CONFIRMATIONS,
                callbackGasLimit: VRF_CALLBACK_GAS_LIMIT,
                numWords: VRF_NUM_WORDS,
                extraArgs: VRFV2PlusClient._argsToBytes(
                    VRFV2PlusClient.ExtraArgsV1({nativePayment: true})
                )
            })
        );
    }

    function fulfillRandomWords(uint256 requestId, uint256[] calldata randomWords) internal {
        if (randomWords.length == 0) revert VrfBadCallback();

        if (requestId != 0 && requestId == vrfHealthRequestId) {
            lastVrfHealthRandomWord = randomWords[0];
            lastVrfHealthFulfilledAt = block.timestamp;
            vrfHealthCheckPassed = true;
            vrfHealthRequestId = 0;
            emit VrfHealthCheckFulfilled(requestId, randomWords[0]);
            return;
        }

        uint256 roundId = vrfRequestIdToRoundId[requestId];
        if (roundId == 0) revert VrfBadCallback();

        Round storage r = rounds[roundId];
        if (Phase(r.phase) != Phase.WAITING_FOR_RANDOMNESS) revert VrfStaleCallback();
        if (r.vrfRequestId != requestId) revert VrfStaleCallback();

        r.randomWord = randomWords[0];
        r.phase = uint8(Phase.RANDOMNESS_FULFILLED);

        delete vrfRequestIdToRoundId[requestId];

        emit PhaseTransition(roundId, Phase.WAITING_FOR_RANDOMNESS, Phase.RANDOMNESS_FULFILLED);
        emit RandomnessFulfilled(roundId, requestId, randomWords[0]);
    }

    function requestRandomnessAgain(uint256 roundId) external nonReentrant {
        Round storage r = rounds[roundId];
        if (Phase(r.phase) != Phase.WAITING_FOR_RANDOMNESS) revert WrongPhase();
        if (block.timestamp < r.vrfRequestedAt + VRF_RETRY_TIMEOUT) revert VrfTooEarly();
        if (r.retryCount >= MAX_CHAINLINK_REQUESTS_PER_ROUND - 1) revert VrfRetryCapReached();

        uint256 bounty = 0;
        if (r.totalRetryBountyPaid + VRF_RETRY_BOUNTY <= MAX_VRF_RETRY_BOUNTY_PER_ROUND) {
            bounty = VRF_RETRY_BOUNTY;
            r.totalRetryBountyPaid += bounty;
        }

        uint256 oldRequestId = r.vrfRequestId;
        delete vrfRequestIdToRoundId[oldRequestId];

        uint256 newRequestId = _requestRandomWords();

        r.vrfRequestId = newRequestId;
        r.vrfRequestedAt = block.timestamp;
        r.retryCount += 1;
        vrfRequestIdToRoundId[newRequestId] = roundId;

        if (bounty > 0) {
            (bool ok, ) = msg.sender.call{value: bounty}("");
            require(ok, "bounty xfer failed");
        }

        emit RandomnessRetried(roundId, msg.sender, newRequestId, bounty);
    }

    /// @notice Compute target, rank top 3, credit pull-payouts, send rake, pay keeper.
    /// v1.1 CHANGE: VRF replenish path branches on shared-sub balance vs
    /// MAX_VRF_RESERVE_BALANCE; overflow (full or partial) credits the winner.
    function finalizeSettlement(uint256 roundId) external nonReentrant {
        Round storage r = rounds[roundId];
        if (Phase(r.phase) != Phase.RANDOMNESS_FULFILLED) revert WrongPhase();

        uint256 pot = r.potTotal;
        uint16 target = uint16(r.randomWord % NUM_SLOTS);

        (uint16 w1, uint16 w2, uint16 w3) = _rankTop3(roundId, target);

        address winner = slotOwner[roundId][w1];
        address second = slotOwner[roundId][w2];
        address third  = slotOwner[roundId][w3];

        r.winnerSlot = w1;
        r.secondSlot = w2;
        r.thirdSlot  = w3;

        uint256 keeperBounty = pot * KEEPER_BPS / BPS_DENOM;
        if (keeperBounty > KEEPER_CAP) keeperBounty = KEEPER_CAP;

        uint256 secondShare  = pot * SECOND_BPS  / BPS_DENOM;
        uint256 thirdShare   = pot * THIRD_BPS   / BPS_DENOM;
        uint256 seedPool     = pot * SEED_BPS    / BPS_DENOM;
        uint256 burnRake     = pot * RAKE_BPS    / BPS_DENOM;
        uint256 founderShare = pot * FOUNDER_BPS / BPS_DENOM;
        uint256 vrfReplenish = pot * VRF_REPLENISH_BPS / BPS_DENOM;
        // v1.1a NEW: throne pool is part of fixedSlices, so winnerShare is
        // automatically reduced by it. Throne payout distribution happens
        // AFTER the standard credit block + season/throne update (see below).
        uint256 thronePool   = pot * THRONE_BPS  / BPS_DENOM;

        uint256 retryBountyOutflows = r.totalRetryBountyPaid;

        uint256 fixedSlices =
            keeperBounty + secondShare + thirdShare + seedPool + burnRake
            + founderShare + vrfReplenish + thronePool + retryBountyOutflows;
        uint256 winnerShare = pot - fixedSlices;

        r.seedDividendPool = seedPool;
        address founderAtSettle = founderPayout;
        r.founderPayoutAtSettle = founderAtSettle;

        // Credit pull-payouts (internal accounting only)
        claimablePayout[roundId][winner] += winnerShare;
        if (second != address(0) && second != winner) {
            claimablePayout[roundId][second] += secondShare;
        } else {
            claimablePayout[roundId][winner] += secondShare;
        }
        if (third != address(0) && third != winner && third != second) {
            claimablePayout[roundId][third] += thirdShare;
        } else {
            claimablePayout[roundId][winner] += thirdShare;
        }
        claimablePayout[roundId][founderAtSettle] += founderShare;

        r.phase = uint8(Phase.SETTLED);

        claimablePayout[roundId][msg.sender] += keeperBounty;

        // ── v1.1a · Throne royalty mechanic (spec §2.6) ──
        // ORDER (operator-locked, dramatic): season check → throne update → throne payout.
        // The crowning round's creator earns their first throne payout on THIS tx.

        // (A) Season reset · uses `>=` so the round settling EXACTLY at the
        //     boundary gets a fresh throne (spec §7 + Codex 2026-05-25 fix).
        //     ThroneLib.wipe + ThroneLib.update operate on the caller's
        //     throne/throneSize storage via delegatecall (EIP-170 size win).
        if (block.timestamp >= uint256(seasonStartedAt) + SEASON_LENGTH) {
            uint64 oldStart = seasonStartedAt;
            ThroneLib.wipe(throne, throneSize);
            seasonStartedAt = uint64(block.timestamp);
            emit SeasonReset(oldStart, seasonStartedAt);
        }

        // (B) Throne update — mutate in place; ThroneSnapshot (below) captures
        //     final post-state for indexer consumption.
        address roundCreator = r.creator;
        if (roundCreator != address(0)) {
            ThroneLib.update(throne, throneSize, roundCreator, pot);
        }

        // (C) Throne payout · ThroneLib.payout distributes 70/20/10 across
        //     non-empty seats; returns the empty-seat overflow for the
        //     caller to route to winner (matches v1.1 VRF-cap-overflow pattern).
        {
            (, uint256 emptySeatOverflow) = ThroneLib.payout(
                throne, thronePool, roundId, claimablePayout
            );
            if (emptySeatOverflow > 0) {
                if (winner == address(0)) revert ZeroWinnerOnOverflow();
                claimablePayout[roundId][winner] += emptySeatOverflow;
            }
        }

        // (D) Throne snapshot · ONE deterministic post-state emission for indexers
        emit ThroneSnapshot(
            roundId,
            throne[0], throneSize[0],
            throne[1], throneSize[1],
            throne[2], throneSize[2]
        );

        // ── v1.1 VRF replenish with cap + overflow-to-winner (spec §3.4) ──
        if (vrfReplenish > 0) {
            // CRITICAL: nativeBalance is the SECOND tuple element of
            // getSubscription (Codex 2026-05-24). LINK balance is first.
            (, uint96 nativeBal, , , ) = s_vrfCoordinator.getSubscription(VRF_SUB_ID);
            uint256 currentNative = uint256(nativeBal);

            if (currentNative >= MAX_VRF_RESERVE_BALANCE) {
                // Full overflow to winner. Defensive: winner is always non-zero
                // (slotOwner of w1, phase invariant guarantees ≥1 owner) but
                // fail loud rather than burn ETH to address(0).
                if (winner == address(0)) revert ZeroWinnerOnOverflow();
                claimablePayout[roundId][winner] += vrfReplenish;
                emit VrfReplenishOverflowToWinner(roundId, winner, vrfReplenish);
            } else if (currentNative + vrfReplenish > MAX_VRF_RESERVE_BALANCE) {
                // Partial: top sub to cap, overflow rest to winner.
                uint256 topUp = MAX_VRF_RESERVE_BALANCE - currentNative;
                uint256 overflowAmt = vrfReplenish - topUp;
                if (winner == address(0)) revert ZeroWinnerOnOverflow();
                // Effects (credit + emit) BEFORE external interaction per CEI.
                claimablePayout[roundId][winner] += overflowAmt;
                emit VrfReplenishOverflowToWinner(roundId, winner, overflowAmt);
                s_vrfCoordinator.fundSubscriptionWithNative{value: topUp}(VRF_SUB_ID);
                emit VrfReserveReplenished(roundId, topUp);
            } else {
                // Normal: full replenish to sub.
                s_vrfCoordinator.fundSubscriptionWithNative{value: vrfReplenish}(VRF_SUB_ID);
                emit VrfReserveReplenished(roundId, vrfReplenish);
            }
        }

        // Send burn rake as plain ETH to $BURN contract · BEST EFFORT.
        (bool rakeOk, ) = BURN_TOKEN.call{value: burnRake}("");
        if (!rakeOk) {
            r.pendingBurnRake = burnRake;
            emit BurnRakeForwardDeferred(roundId, burnRake);
        }

        emit PhaseTransition(roundId, Phase.RANDOMNESS_FULFILLED, Phase.SETTLED);
        emit RoundSettled(
            roundId,
            winner,
            w1,
            target,
            pot,
            winnerShare,
            second,
            secondShare,
            third,
            thirdShare,
            seedPool,
            burnRake,
            founderAtSettle,
            founderShare,
            keeperBounty,
            msg.sender
        );
    }

    /// @notice Retry a deferred BURN rake transfer. Permissionless.
    /// NOT gated by !deprecated: pre-deprecation SETTLED rounds with pending
    /// rake can still be flushed by anyone after deprecation. (sub-agent 2026-05-24)
    function forwardPendingBurnRake(uint256 roundId) external nonReentrant {
        Round storage r = rounds[roundId];
        uint256 amount = r.pendingBurnRake;
        if (amount == 0) revert NothingToWithdraw();
        r.pendingBurnRake = 0;
        (bool ok, ) = BURN_TOKEN.call{value: amount}("");
        require(ok, "rake xfer still failed");
        emit BurnRakeForwarded(roundId, amount);
    }

    function _rankTop3(uint256 roundId, uint16 target)
        internal
        view
        returns (uint16 first, uint16 second, uint16 third)
    {
        uint16[] storage slots = _claimedSlots[roundId];
        uint256 n = slots.length;
        if (n == 0) revert WrongPhase();

        first = slots[0];
        uint32 d1 = _distance(first, target);

        uint32 d2 = type(uint32).max;
        uint32 d3 = type(uint32).max;

        for (uint256 i = 1; i < n; i++) {
            uint16 s = slots[i];
            uint32 d = _distance(s, target);

            if (d < d1 || (d == d1 && s < first)) {
                d3 = d2; third = second;
                d2 = d1; second = first;
                d1 = d;  first = s;
            } else if (d < d2 || (d == d2 && s < second)) {
                d3 = d2; third = second;
                d2 = d;  second = s;
            } else if (d < d3 || (d == d3 && s < third)) {
                d3 = d;  third = s;
            }
        }
    }

    function _distance(uint16 a, uint16 b) internal pure returns (uint32) {
        return a > b ? uint32(a) - uint32(b) : uint32(b) - uint32(a);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  v1.1a · Throne view (mutation logic lives in ThroneLib)
    // ─────────────────────────────────────────────────────────────────────

    /// @notice Throne state + season info. `stale=true` indicates the stored
    /// throne corresponds to an expired season; the next finalizeSettlement
    /// will wipe it. UIs should render "season expired — fresh throne on next
    /// round" when stale==true rather than showing the stale seats.
    function getThrone()
        external
        view
        returns (
            address[3] memory seats,
            uint256[3] memory sizes,
            uint64 seasonStart,
            uint64 seasonEnd,
            bool stale
        )
    {
        seats = throne;
        sizes = throneSize;
        seasonStart = seasonStartedAt;
        seasonEnd = seasonStartedAt + uint64(SEASON_LENGTH);
        stale = block.timestamp >= uint256(seasonEnd);
    }

    function vrfReserveBalance() public view returns (uint256) {
        (, uint96 nativeBalance, , , ) = s_vrfCoordinator.getSubscription(VRF_SUB_ID);
        return uint256(nativeBalance);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Withdrawals (pull-payment everywhere)
    // ─────────────────────────────────────────────────────────────────────

    function withdrawPayout(uint256 roundId) external nonReentrant {
        _withdrawPayoutTo(roundId, payable(msg.sender));
    }

    function withdrawPayoutTo(uint256 roundId, address payable recipient) external nonReentrant {
        if (recipient == address(0)) revert BurnBomb_ZeroAddress();
        _withdrawPayoutTo(roundId, recipient);
    }

    function _withdrawPayoutTo(uint256 roundId, address payable recipient) internal {
        uint256 amount = claimablePayout[roundId][msg.sender];
        if (amount == 0) revert NothingToWithdraw();

        claimablePayout[roundId][msg.sender] = 0;

        (bool ok, ) = recipient.call{value: amount}("");
        require(ok, "withdraw failed");

        emit PayoutWithdrawn(roundId, msg.sender, recipient, amount);
    }

    function withdrawSeedDividend(uint256 roundId) external nonReentrant {
        Round storage r = rounds[roundId];
        if (Phase(r.phase) != Phase.SETTLED) revert WrongPhase();
        if (seedDividendClaimed[roundId][msg.sender]) revert AlreadyClaimed();

        uint256 contrib = seedContribution[roundId][msg.sender];
        if (contrib == 0) revert NothingToWithdraw();

        uint256 amount = r.seedDividendPool * contrib / r.seedPotTotal;
        if (amount == 0) revert NothingToWithdraw();

        seedDividendClaimed[roundId][msg.sender] = true;

        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "seed div xfer failed");

        emit SeedDividendWithdrawn(roundId, msg.sender, amount);
    }

    function withdrawRefund(uint256 roundId, uint16 slotIndex) external nonReentrant {
        Round storage r = rounds[roundId];
        if (Phase(r.phase) != Phase.REFUNDED) revert WrongPhase();
        if (slotOwner[roundId][slotIndex] != msg.sender) revert NotSlotOwner();

        uint256 grossAmount = slotValue(slotIndex);
        uint256 amount = grossAmount;
        if (r.totalRetryBountyPaid > 0 && r.potTotal > 0) {
            uint256 shortfall = grossAmount * r.totalRetryBountyPaid / r.potTotal;
            amount = grossAmount - shortfall;
        }

        slotOwner[roundId][slotIndex] = address(0);

        uint256 contractBal = address(this).balance;
        if (amount > contractBal) amount = contractBal;

        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "refund xfer failed");

        emit RefundClaimed(roundId, msg.sender, slotIndex, amount);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  v1.1 NEW · Bulk withdraw functions (spec §5)
    //
    //  All three: nonReentrant + effects-before-interactions + single
    //  per-batch ETH transfer at the end. MAX_BATCH_SIZE cap guards
    //  against OOG even though Minnow's NUM_SLOTS naturally bounds refund.
    // ─────────────────────────────────────────────────────────────────────

    /// @notice Batch-claim refunds for multiple slots in a single REFUNDED round.
    /// @dev Atomic: any wrong-owner slot reverts the entire batch.
    function withdrawRefundBatch(uint256 roundId, uint16[] calldata slotIndices)
        external
        nonReentrant
    {
        Round storage r = rounds[roundId];
        if (Phase(r.phase) != Phase.REFUNDED) revert WrongPhase();
        if (slotIndices.length == 0) revert EmptyBatch();
        if (slotIndices.length > MAX_BATCH_SIZE) revert BatchTooLarge();

        uint256 total;
        uint256 contractBal = address(this).balance;
        // Cache pro-rata inputs once — they're read-only during the loop.
        uint256 totalBounty = r.totalRetryBountyPaid;
        uint256 potTotal = r.potTotal;

        for (uint256 i = 0; i < slotIndices.length; i++) {
            uint16 slotIndex = slotIndices[i];
            // Wrong-owner — atomic revert. Also naturally rejects a duplicate
            // index in the same call (second visit sees address(0) ≠ msg.sender).
            if (slotOwner[roundId][slotIndex] != msg.sender) revert NotSlotOwner();

            uint256 grossAmount = slotValue(slotIndex);
            uint256 amount = grossAmount;
            if (totalBounty > 0 && potTotal > 0) {
                uint256 shortfall = grossAmount * totalBounty / potTotal;
                amount = grossAmount - shortfall;
            }

            // Effects: clear ownership BEFORE accumulating (per-slot atomic).
            slotOwner[roundId][slotIndex] = address(0);

            // Defensive: rounding dust could push the running total past
            // contract balance; cap per-slot so the final transfer fits.
            if (total + amount > contractBal) {
                amount = contractBal > total ? contractBal - total : 0;
            }
            total += amount;

            emit RefundClaimed(roundId, msg.sender, slotIndex, amount);
        }

        (bool ok, ) = msg.sender.call{value: total}("");
        require(ok, "refund batch xfer failed");
    }

    /// @notice Batch-claim payouts across multiple SETTLED rounds.
    /// @dev Zero-claimable rounds silently skip; phase check is per-round
    /// strict (any non-SETTLED round in the array reverts the batch).
    function withdrawPayoutBatch(uint256[] calldata roundIds) external nonReentrant {
        if (roundIds.length == 0) revert EmptyBatch();
        if (roundIds.length > MAX_BATCH_SIZE) revert BatchTooLarge();

        uint256 total;
        for (uint256 i = 0; i < roundIds.length; i++) {
            uint256 roundId = roundIds[i];
            Round storage r = rounds[roundId];
            if (Phase(r.phase) != Phase.SETTLED) revert WrongPhase();

            uint256 amount = claimablePayout[roundId][msg.sender];
            if (amount > 0) {
                claimablePayout[roundId][msg.sender] = 0;
                total += amount;
                // PayoutWithdrawn v1 signature: (roundId, account, recipient, amount).
                // In batch, account == recipient == msg.sender.
                emit PayoutWithdrawn(roundId, msg.sender, msg.sender, amount);
            }
        }

        if (total == 0) revert NothingToWithdraw();
        (bool ok, ) = msg.sender.call{value: total}("");
        require(ok, "payout batch xfer failed");
    }

    /// @notice Batch-claim seed dividends across multiple SETTLED rounds.
    /// @dev Already-claimed or zero-contribution rounds silently skip.
    function withdrawSeedDividendBatch(uint256[] calldata roundIds) external nonReentrant {
        if (roundIds.length == 0) revert EmptyBatch();
        if (roundIds.length > MAX_BATCH_SIZE) revert BatchTooLarge();

        uint256 total;
        for (uint256 i = 0; i < roundIds.length; i++) {
            uint256 roundId = roundIds[i];
            Round storage r = rounds[roundId];
            if (Phase(r.phase) != Phase.SETTLED) revert WrongPhase();

            if (seedDividendClaimed[roundId][msg.sender]) continue;
            uint256 contribution = seedContribution[roundId][msg.sender];
            if (contribution == 0) continue;

            uint256 amount = r.seedDividendPool * contribution / r.seedPotTotal;
            if (amount > 0) {
                seedDividendClaimed[roundId][msg.sender] = true;
                total += amount;
                emit SeedDividendWithdrawn(roundId, msg.sender, amount);
            }
        }

        if (total == 0) revert NothingToWithdraw();
        (bool ok, ) = msg.sender.call{value: total}("");
        require(ok, "seed div batch xfer failed");
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Founder payout · two-step rotation (survives freezeAdmin)
    // ─────────────────────────────────────────────────────────────────────

    function proposeFounderPayoutAddress(address newAddress) external onlyOwner {
        if (newAddress == address(0)) revert BurnBomb_ZeroAddress();
        pendingFounderPayout = newAddress;
        emit FounderPayoutProposed(founderPayout, newAddress);
    }

    function acceptFounderPayoutAddress() external {
        address proposed = pendingFounderPayout;
        if (proposed == address(0)) revert NoPendingProposal();
        if (msg.sender != proposed) revert NotProposedRecipient();

        address old = founderPayout;
        founderPayout = proposed;
        pendingFounderPayout = address(0);

        emit FounderPayoutAccepted(old, proposed);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Lifecycle: activate + deprecate
    // ─────────────────────────────────────────────────────────────────────

    function activate() external onlyOwner notDeprecated {
        if (activated) revert AlreadyActivated();
        if (vrfReserveBalance() < MIN_VRF_RESERVE_TO_ACTIVATE) revert VrfReserveTooLow();
        if (!vrfHealthCheckPassed) revert VrfHealthCheckNotPassed();
        activated = true;
        // v1.1a: anchor the throne season clock at the operational launch moment.
        // Spec §7.2: NOT at deploy time (deploy → activate gap could be days).
        seasonStartedAt = uint64(block.timestamp);
        emit Activated(msg.sender, block.timestamp);
    }

    function deprecate() external onlyOwner {
        if (deprecated) revert AlreadyDeprecated();
        deprecated = true;
        emit Deprecated(msg.sender, block.timestamp);
    }

    /// @notice v1.1 NEW · Recover VRF sub balance after graceful retire.
    /// Triple-gated: must be deprecated AND current round terminal AND no
    /// pending VRF request. Carries `adminNotFrozen` so recovery must happen
    /// BEFORE freezeAdmin (intentional runbook).
    /// Canonical runbook: deprecate() → wait for current round to
    /// SETTLED/REFUNDED → confirm pendingRequestExists == false →
    /// cancelVrfSubscription(recipient) → optionally freezeAdmin afterward.
    function cancelVrfSubscription(address payable recipient)
        external
        onlyOwner
        adminNotFrozen
    {
        if (!deprecated) revert ContractNotDeprecated();
        if (recipient == address(0)) revert BurnBomb_ZeroAddress();

        // Live-round guard (Codex 2026-05-24): deprecated alone doesn't stop
        // in-flight rounds; cancelling VRF mid-round would strand settlement.
        if (currentRoundId != 0) {
            Phase cur = Phase(rounds[currentRoundId].phase);
            if (cur != Phase.SETTLED && cur != Phase.REFUNDED) revert ActiveRoundExists();
        }

        // Defence-in-depth (Codex 2026-05-24): catch any in-flight request
        // even if the round phase check passed (shouldn't happen given the
        // phase guard, but cheap to verify against Chainlink directly).
        if (s_vrfCoordinator.pendingRequestExists(VRF_SUB_ID)) {
            revert VrfRequestPending();
        }

        // Read native balance for event emission BEFORE cancellation invalidates
        // the sub. (Coordinator hands the balance to recipient internally.)
        (, uint96 nativeBal, , , ) = s_vrfCoordinator.getSubscription(VRF_SUB_ID);

        s_vrfCoordinator.cancelSubscription(VRF_SUB_ID, recipient);

        emit VrfSubscriptionCancelled(recipient, uint256(nativeBal));
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Admin (revocable via freezeAdmin)
    // ─────────────────────────────────────────────────────────────────────

    function pause() external onlyOwner adminNotFrozen {
        _pause();
    }

    function unpause() external onlyOwner adminNotFrozen {
        _unpause();
    }

    function emergencyResolveStuckRound(uint256 roundId) external onlyOwner {
        Round storage r = rounds[roundId];
        Phase p = Phase(r.phase);
        if (p == Phase.SETTLED || p == Phase.REFUNDED) revert WrongPhase();

        uint256 emergencyStart;
        if (p == Phase.WAITING_FOR_RANDOMNESS || p == Phase.RANDOMNESS_FULFILLED) {
            emergencyStart = r.firstVrfRequestedAt;
        } else {
            emergencyStart = uint256(r.createdAt);
        }
        if (block.timestamp < emergencyStart + EMERGENCY_REFUND_DELAY) revert EmergencyTooEarly();

        r.phase = uint8(Phase.REFUNDED);
        emit EmergencyRefundEnabled(roundId);
        emit RoundRefunded(roundId);
    }

    /// @notice OVERRIDE · renounceOwnership disabled. v1's ownership came from
    /// Chainlink's ConfirmedOwner which did not expose renounce. v1.1 switched
    /// to OZ Ownable2Step which DOES — explicit override reverts to preserve
    /// the v1 behavior + operator's "never renounce" runbook (per memory
    /// `project_burntoken_operational_model.md`). freezeAdmin is the intended
    /// way to lock admin capabilities, not renounce.
    /// (Codex 2026-05-24 re-audit pass.)
    function renounceOwnership() public pure override {
        revert AdminIsFrozen();
    }

    function freezeAdmin() external onlyOwner {
        if (adminFrozen) revert AlreadyFrozen();
        if (paused()) revert CannotFreezeWhilePaused();
        adminFrozen = true;
        emit AdminFrozen(msg.sender);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Views
    // ─────────────────────────────────────────────────────────────────────

    function getRound(uint256 roundId) external view returns (Round memory) {
        return rounds[roundId];
    }

    function getRoundTiming(uint256 roundId)
        external
        view
        returns (Phase phase, uint256 nextDeadline)
    {
        Round storage r = rounds[roundId];
        phase = Phase(r.phase);
        if (phase == Phase.SEEDING) nextDeadline = r.seedEndsAt;
        else if (phase == Phase.ACTIVE) {
            uint256 a = r.activeTimerEndsAt;
            uint256 b = r.activeEndsAt;
            nextDeadline = a < b ? a : b;
        }
        else if (phase == Phase.FINAL_COUNTDOWN) nextDeadline = r.finalCountdownEndsAt;
        else nextDeadline = 0;
    }

    function getClaimedSlots(uint256 roundId) external view returns (uint16[] memory) {
        return _claimedSlots[roundId];
    }

    function getAddressSlots(uint256 roundId, address player) external view returns (uint16[] memory) {
        return addressSlots[roundId][player];
    }

    // isSlotAvailable() view removed in v1.1a · derivable client-side from
    // slotOwner(roundId, slotIndex) == address(0) + slotIndex < NUM_SLOTS().

    // NOTE · v1.1a: previewSeedDividend() and previewRefund() views from v1.1
    // are REMOVED to fit EIP-170. Both are deterministic functions of raw
    // public state — frontends compute them client-side:
    //   previewSeedDividend = r.seedDividendPool * seedContribution[r][a] / r.seedPotTotal
    //     (only meaningful when r.phase == SETTLED + !seedDividendClaimed[r][a])
    //   previewRefund = slotValue(i) - (slotValue(i) * r.totalRetryBountyPaid / r.potTotal)
    //     (only meaningful when r.phase == REFUNDED + slotOwner[r][i] != address(0))
    // All inputs (rounds(r), seedContribution, seedDividendClaimed, slotOwner)
    // remain readable via the auto-generated public getters.

    // ─────────────────────────────────────────────────────────────────────
    //  ETH handling · reject direct sends
    // ─────────────────────────────────────────────────────────────────────

    receive() external payable {
        revert DirectSendsRejected();
    }

    fallback() external payable {
        revert DirectSendsRejected();
    }
}


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
